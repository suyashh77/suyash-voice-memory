"""Meaningful storage, recovery, capture, and desktop smoke checks. No microphone recording."""
import json
import os
import queue
import tempfile
import threading
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memory_store import ConflictError, Store
from speech import Recorder, import_segments


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def note(self, body='The microphone is local.', **kw):
        return self.store.create(kw.pop('project', 'Voice'), 'note', kw.pop('title', 'A note'), body, **kw)

    def test_private_and_proposed_do_not_leak(self):
        private = self.note('secret cactus')
        proposed = self.note('suggested cactus', status='proposed')
        shared = self.note('shared cactus', status='confirmed')
        self.assertEqual([x['id'] for x in self.store.list(query='cactus', shared_only=True)], [shared['id']])
        for item in (private, proposed):
            with self.assertRaises(ValueError):
                self.store.get(item['id'], shared_only=True)
        self.store.update(proposed['id'], 1, status='confirmed')
        self.assertEqual(len(self.store.list(query='cactus', shared_only=True)), 2)

    def test_stale_writes_fail_and_history_survives(self):
        item = self.note('before')
        self.store.update(item['id'], 1, body='after')
        with self.assertRaises(ConflictError):
            self.store.update(item['id'], 1, body='stale')
        self.assertEqual(self.store.get(item['id'])['body'], 'after')
        with self.store.connect() as c:
            history = c.execute('SELECT snapshot FROM history WHERE memory_id=? ORDER BY revision', (item['id'],)).fetchall()
        self.assertEqual([json.loads(r[0])['body'] for r in history], ['before', 'after'])

    def test_concurrent_clients_cannot_silently_overwrite(self):
        item = self.note()
        def update(n):
            try:
                self.store.update(item['id'], 1, body=str(n))
                return True
            except ConflictError:
                return False
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(update, range(4)))
        self.assertEqual(sum(results), 1)
        current = self.store.get(item['id'])
        self.assertIn(current['body'], (self.store.root / 'memory' / (item['id']+'.md')).read_text(encoding='utf-8'))

    def test_search_handles_names_punctuation_and_project_scope(self):
        self.note('Suyash uses Bengaluru names.', status='confirmed', project='Personal')
        self.note('Voice application', status='confirmed', project='Voice')
        self.assertEqual(len(self.store.list(query='"Suyash" OR Bengaluru!')), 1)
        self.assertEqual(self.store.list(project='Voice', query='Bengaluru'), [])

    def test_handoff_has_provenance_and_excludes_private(self):
        self.note('Private password: do not export')
        self.note('Wants voice input', project='Personal', status='confirmed')
        self.note('Use local speech', status='confirmed', source='User instruction')
        path, text = self.store.handoff('Voice')
        self.assertNotIn('Private password', text)
        self.assertIn('Wants voice input', text)
        self.assertIn('User instruction', text)
        self.assertEqual(path.read_text(encoding='utf-8'), text)

    def session_audio(self, sequence=0, count=1600):
        item = self.store.create('Voice', 'transcript', 'Test recording')
        folder = self.store.audio / item['id']
        folder.mkdir()
        path = folder / f'{sequence:06d}.wav'
        with wave.open(str(path), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b'\0\0' * count)
        return item, path

    def test_audio_recovery_is_idempotent(self):
        item, path = self.session_audio()
        self.assertEqual(self.store.recover_audio(), 1)
        self.assertEqual(self.store.recover_audio(), 0)
        self.assertEqual(self.store.chunks(pending=True)[0]['session_id'], item['id'])

    def test_correction_retains_original_audio_and_labels(self):
        item, path = self.session_audio()
        chunk = self.store.add_chunk(item['id'], 0, path, .1)
        self.store.finish_chunk(chunk, 'Sue yash likes notes', 'test-model')
        self.store.correct(chunk, 'Suyash likes notes')
        self.assertEqual(self.store.get(item['id'])['body'], 'Suyash likes notes')
        raw = self.store.chunks(item['id'])[0]
        self.assertEqual(raw['original'], 'Sue yash likes notes')
        with self.store.connect() as c:
            correction = c.execute('SELECT * FROM corrections').fetchone()
        self.assertEqual(correction['audio_path'], str(path))
        self.assertEqual(correction['corrected'], 'Suyash likes notes')

    def test_failed_segments_can_retry_without_losing_audio(self):
        item, path = self.session_audio()
        chunk = self.store.add_chunk(item['id'], 0, path, .1)
        self.store.finish_chunk(chunk, '', 'test', 'temporary failure')
        self.assertTrue(path.exists())
        self.store.retry()
        self.assertEqual(self.store.chunks(pending=True)[0]['id'], chunk)

    def test_overlap_deduplicates_only_overlapping_segments(self):
        item, path = self.session_audio()
        first = self.store.add_chunk(item['id'], 0, path, .1)
        second = self.store.add_chunk(item['id'], 1, path.with_name('000001.wav'), .1, overlap=.8)
        self.store.finish_chunk(first, 'We should build shared memory.', 'test')
        self.store.finish_chunk(second, 'shared memory for all my models.', 'test')
        self.assertEqual(self.store.get(item['id'])['body'], 'We should build shared memory.\n\nfor all my models.')

    def test_capture_writer_flushes_final_short_piece(self):
        item = self.store.create('Voice', 'transcript', 'Capture')
        recorder = Recorder(self.store, item['id'])
        recorder.rate = 16000
        recorder.running = True
        recorder.writer = threading.Thread(target=recorder._write)
        recorder.writer.start()
        recorder.frames.put(b'\1\0' * 1600)
        recorder.frames.put(b'\1\0' * 1600)
        recorder.stop()
        row = self.store.chunks(item['id'])[0]
        with wave.open(row['path'], 'rb') as audio:
            self.assertEqual(audio.getnframes(), 3200)
        self.assertAlmostEqual(row['duration'], .2)

    def test_long_capture_splits_on_disk_instead_of_holding_everything(self):
        item = self.store.create('Voice', 'transcript', 'Long capture')
        recorder = Recorder(self.store, item['id'])
        recorder.rate = 16000
        recorder.running = True
        recorder.writer = threading.Thread(target=recorder._write)
        recorder.writer.start()
        # Constant non-silent synthetic input forces overlap after 25 seconds.
        for _ in range(270):
            recorder.frames.put(b'\xff\x0f' * 1600)
        recorder.stop()
        rows = self.store.chunks(item['id'])
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[1]['overlap'], .8)
        self.assertAlmostEqual(sum(r['duration'] - r['overlap'] for r in rows), 27)

    def test_desktop_builds_and_can_show_shared_memory(self):
        import tkinter as tk
        from app import App
        root = tk.Tk()
        root.withdraw()
        app = App(root, self.store, start_worker=False)
        item = self.note('A real shared note', status='confirmed')
        app.refresh_memory()
        self.assertTrue(app.tree.exists(item['id']))
        app.load_memory(item)
        self.assertIn('A real shared note', app.note_body.get('1.0', 'end'))
        root.update_idletasks()
        root.destroy()

    def test_audio_import_resamples_and_splits_incrementally(self):
        item, source = self.session_audio(count=26 * 16000)
        imported = self.store.create('Voice', 'transcript', 'Imported')
        count = import_segments(source, self.store, imported['id'])
        self.assertEqual(count, 2)
        chunks = self.store.chunks(imported['id'])
        self.assertAlmostEqual(sum(c['duration'] - c['overlap'] for c in chunks), 26)
        with wave.open(chunks[0]['path'], 'rb') as audio:
            self.assertEqual(audio.getframerate(), 16000)
            self.assertEqual(audio.getnchannels(), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
