"""Microphone capture and offline transcription; microphone opens only on Start."""
from __future__ import annotations

import math
import os
import queue
import threading
import time
import wave
from array import array
from pathlib import Path

from memory_store import ROOT
from adaptation import apply_rules
from recognition_config import decoder_hints, load_config

def import_segments(source, store, session):
    """Decode large imports incrementally; keep at most one short piece in memory."""
    import av
    folder = store.audio / session
    folder.mkdir(exist_ok=True)
    rate, size, overlap_bytes = 16000, 25 * 16000 * 2, int(.8 * 16000) * 2
    buffer = bytearray()
    sequence = 0
    def save(block):
        nonlocal sequence
        out = folder / f'{sequence:06d}.wav'
        with wave.open(str(out), 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(block)
        store.add_chunk(session, sequence, out, len(block)/(2*rate), .8 if sequence else 0)
        sequence += 1
    with av.open(str(source), mode='r') as container:
        resampler = av.AudioResampler(format='s16', layout='mono', rate=rate)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                buffer.extend(converted.to_ndarray().tobytes())
                while len(buffer) >= size:
                    save(bytes(buffer[:size]))
                    del buffer[:size-overlap_bytes]
        for converted in resampler.resample(None):
            buffer.extend(converted.to_ndarray().tobytes())
        if len(buffer) > (overlap_bytes if sequence else 0):
            save(bytes(buffer))
    return sequence


class Recorder:
    def __init__(self, store, session, device=None, notify=lambda *a: None):
        self.store, self.session, self.device = store, session, device
        self.notify = notify
        self.frames = queue.Queue(maxsize=600)  # one minute buffer; visible error on overflow
        self.error = None
        self.level = 0.0
        self.stream = None
        self.writer = None
        self.running = False
        self.started = None

    def start(self):
        import sounddevice as sd
        info = sd.query_devices(self.device, 'input')
        self.rate = int(info['default_samplerate'])
        sd.check_input_settings(device=self.device, channels=1, dtype='int16', samplerate=self.rate)
        self.stream = sd.RawInputStream(device=self.device, samplerate=self.rate,
                    channels=1, dtype='int16', blocksize=max(1, self.rate // 10), callback=self._callback)
        self.running = True
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()
        try:
            self.stream.start()
            self.started = time.monotonic()
        except Exception:
            self.stop()
            raise

    def _callback(self, data, frames, timing, status):
        if status:
            self.error = 'Microphone reported: ' + str(status)
        if not self.running:
            return
        try:
            self.frames.put_nowait(bytes(data))
        except queue.Full:
            self.error = 'Audio capture could not keep up. Recording stopped; saved audio is safe.'
            self.running = False

    def _write(self):
        folder = self.store.audio / self.session
        folder.mkdir(exist_ok=True)
        previous = self.store.chunks(self.session)
        sequence = max((r['sequence'] for r in previous), default=-1) + 1
        audio = None
        file = None
        count = 0
        quiet = 0.0
        tail = b''
        carry = b''
        overlap = 0.0
        try:
            from streaming_vad import StreamingVAD
            vad = StreamingVAD(self.rate)
        except Exception as exc:
            vad = None
            self.notify('status', f'Neural pause detector unavailable ({exc}); saving fixed 25-second pieces.')
        try:
            while True:
                block = self.frames.get()
                if block is None:
                    break
                if audio is None:
                    path = folder / f'{sequence:06d}.wav'
                    file = open(path, 'wb')
                    audio = wave.open(file, 'wb')
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(self.rate)
                    overlap = len(carry) / (2 * self.rate)
                    if carry:
                        audio.writeframes(carry)
                        count = len(carry) // 2
                        carry = b''
                samples = array('h', block)
                rms = math.sqrt(sum(v*v for v in samples) / max(len(samples), 1)) / 32768
                self.level = min(rms * 8, 1.0)
                # writeframes updates the WAV header so interrupted files remain recoverable.
                audio.writeframes(block)
                file.flush()
                count += len(samples)
                if vad is not None:
                    try:
                        quiet = vad.feed(block)
                    except Exception as exc:
                        vad = None
                        quiet = 0
                        self.notify('status', f'Pause detector failed ({exc}); using fixed pieces. Audio remains saved.')
                tail = (tail + block)[-int(self.rate * .8) * 2:]
                seconds = count / self.rate
                if (seconds >= 3 and quiet >= .7) or seconds >= 25:
                    audio.close()
                    file.close()
                    audio = file = None
                    self.store.add_chunk(self.session, sequence, path, seconds, overlap)
                    self.notify('chunk', self.session)
                    carry = tail if quiet < .7 else b''
                    tail = b''
                    sequence += 1
                    count = 0
                    quiet = 0
        except Exception as exc:
            self.error = f'Audio could not be saved: {exc}'
            self.running = False
        finally:
            if audio is not None:
                audio.close()
                file.close()
                if count:
                    try:
                        self.store.add_chunk(self.session, sequence, path, count / self.rate, overlap)
                        self.notify('chunk', self.session)
                    except Exception as exc:
                        self.error = f'Audio saved; indexing needs recovery: {exc}'

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None
        if self.writer and self.writer.is_alive():
            while self.writer.is_alive():
                try:
                    self.frames.put(None, timeout=.2)
                    break
                except queue.Full:
                    pass
            self.writer.join(timeout=10)
            if self.writer.is_alive():
                raise RuntimeError('Audio is still being saved. Please wait before closing.')


class Transcriber:
    def __init__(self, store, notify=lambda *a: None):
        self.store, self.notify = store, notify
        self.model = None
        self.ready = False
        self.failure = None
        self.busy = False
        self.stopping = threading.Event()
        self.thread = None
        self.config = load_config(store.root)
        self.device = 'cpu'
        self.model_path = ROOT / 'models' / self.config['model']
        self.last_applied = []

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def load(self):
        # Never downloads on startup or sends recordings anywhere.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        from faster_whisper import WhisperModel
        if not (self.model_path / 'model.bin').exists():
            raise RuntimeError('Speech model missing. Run Setup Voice.cmd once while online.')
        import ctranslate2
        self.device = 'cpu'
        if self.config['device'] in ('auto', 'cuda'):
            try:
                if ctranslate2.get_cuda_device_count() > 0:
                    self.device = 'cuda'
            except Exception as exc:
                self.notify('status', f'GPU unavailable: {exc}. Using CPU.')
        try:
            self._load_device(self.device)
        except Exception:
            if self.device == 'cpu':
                raise
            self.notify('status', 'GPU could not load the model; using CPU.')
            self._load_device('cpu')
        self.ready = True

    def _load_device(self, device):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(str(self.model_path), device=device,
                                 compute_type='int8_float16' if device == 'cuda' else 'int8',
                                 cpu_threads=min(8, max(2, (os.cpu_count() or 4) // 2)),
                                 num_workers=1, local_files_only=True)
        self.device = device

    @property
    def label(self):
        return f'{self.config["model"]} / {self.device} / beam {self.config["beam_size"]}'

    def transcribe(self, path, session=None, sequence=0, personalization=None):
        enabled = self.config['personalization'] if personalization is None else personalization
        context = self.store.recognition_context(session, sequence) if enabled else {'hints': [], 'previous': '', 'project': ''}
        previous = context['previous'] if self.config['previous_context'] else ''
        prompt, hints = decoder_hints(self.model.hf_tokenizer, context['hints'], previous)
        segments, info = self.model.transcribe(str(path), language=self.config['language'], beam_size=self.config['beam_size'],
                    vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
                    initial_prompt=prompt if enabled else None, hotwords=hints if enabled else None,
                    condition_on_previous_text=False)
        self.last_raw = ' '.join(segment.text.strip() for segment in segments).strip()
        rules = self.store.adaptation_rules(context['project']) if enabled else []
        adapted, self.last_applied = apply_rules(self.last_raw, rules)
        return adapted

    def process_chunk(self, chunk):
        try:
            text = self.transcribe(chunk['path'], chunk['session_id'], chunk['sequence'])
        except Exception:
            # CUDA can fail lazily while the transcription generator is consumed.
            if self.device != 'cuda':
                raise
            self.model = None
            import gc
            gc.collect()
            self._load_device('cpu')
            self.notify('status', 'GPU transcription failed; retried on CPU.')
            text = self.transcribe(chunk['path'], chunk['session_id'], chunk['sequence'])
        self.store.finish_chunk(chunk['id'], self.last_raw, self.label,
                                adapted=text, applied_rules=self.last_applied)
        return text

    def _run(self):
        try:
            self.notify('status', 'Loading local speech model…')
            self.load()
            self.notify('status', f'Ready · {self.label} · speech stays on this computer')
        except Exception as exc:
            self.failure = str(exc)
            self.notify('error', self.failure)
            return
        while not self.stopping.is_set():
            pending = self.store.chunks(pending=True)
            if not pending:
                self.busy = False
                self.stopping.wait(.5)
                continue
            self.busy = True
            for chunk in pending:
                if self.stopping.is_set():
                    break
                try:
                    self.process_chunk(chunk)
                    self.notify('transcribed', chunk['session_id'])
                except Exception as exc:
                    self.store.finish_chunk(chunk['id'], '', self.label, str(exc))
                    self.notify('error', f'Audio saved, transcription failed: {exc}')

    def stop(self):
        self.stopping.set()
