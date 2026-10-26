"""Export verified audio locally. No model-generated reference labels, no uploads."""
import argparse
import hashlib
import json
import wave
from pathlib import Path
from memory_store import DATA, Store


def export_dataset(store, destination):
    import av
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    records, hashes = [], {}
    with store.connect() as c:
        rows = c.execute('''SELECT c.*,m.project FROM chunks c JOIN memories m ON m.id=c.session_id
                 WHERE c.verified=1 AND c.status='done' ORDER BY c.session_id,c.sequence''').fetchall()
    if not rows:
        raise ValueError('No verified audio. Listen to a segment, save its correction or verify its exact words first.')
    try:
        for row in rows:
            text = row['corrected'] if row['corrected'] is not None else row['original']
            if not text.strip():
                continue
            source = Path(row['path']).resolve()
            if not source.is_relative_to(store.audio.resolve()):
                raise ValueError('Audio must be inside this memory library: ' + row['id'])
            samples = []
            with av.open(str(source)) as container:
                resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
                for frame in container.decode(audio=0):
                    samples += [f.to_ndarray().tobytes() for f in resampler.resample(frame)]
                samples += [f.to_ndarray().tobytes() for f in resampler.resample(None)]
            pcm = b''.join(samples)
            duration = len(pcm) / 32000
            if not 0 < duration <= 30:
                raise ValueError('Training chunks must be at most 30 seconds: ' + row['id'])
            digest = hashlib.sha256(pcm).hexdigest()
            if digest in hashes:
                if hashes[digest] != row['dataset_split']:
                    raise ValueError('Identical audio appears in different splits. Keep duplicates in one split.')
                continue
            hashes[digest] = row['dataset_split']
            out = destination / (row['id'] + '.wav')
            with wave.open(str(out), 'wb') as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(pcm)
            records.append(dict(id=row['id'], session_id=row['session_id'], audio=out.name,
                                text=text, split=row['dataset_split'], duration=duration,
                                audio_sha256=digest, original=row['original'], project=row['project']))
        sessions = {}
        for record in records:
            sessions.setdefault(record['session_id'], set()).add(record['split'])
        if any(len(splits) != 1 for splits in sessions.values()):
            raise ValueError('A session crosses dataset splits. Assign whole sessions before export.')
        (destination / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
        summary = {split: {'segments': sum(r['split'] == split for r in records),
                          'minutes': round(sum(r['duration'] for r in records if r['split'] == split)/60, 2)}
                   for split in ('train','validation','test')}
        (destination / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary
    except Exception:
        # Leave inspectable local output on failure, but no manifest that could be trained accidentally.
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(export_dataset(Store(args.data), args.destination), indent=2))
