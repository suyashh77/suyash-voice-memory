"""Compare plain and personalized decoding on explicitly held-out verified audio."""
import argparse
import json
import re
import time
from pathlib import Path
from memory_store import DATA, ROOT, Store
from speech import Transcriber


def word_errors(reference, hypothesis):
    normalize = lambda text: re.findall(r"\w+(?:'\w+)?", text.casefold())
    ref, hyp = normalize(reference), normalize(hypothesis)
    previous = list(range(len(hyp)+1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1], len(ref)


def run(store, model, split='test', model_dir=None, device='cpu'):
    with store.connect() as c:
        rows = [dict(r) for r in c.execute('''SELECT * FROM chunks WHERE verified=1
                AND status='done' AND dataset_split=? ORDER BY session_id,sequence''', (split,))]
    if not rows:
        raise ValueError(f'No verified {split} audio. Reserve a separate session and verify its exact words first.')
    t = Transcriber(store)
    t.config.update(model=model, device=device, previous_context=False)
    t.model_path = Path(model_dir) if model_dir else ROOT / 'models' / model
    t.load()
    totals = {mode: {'errors': 0, 'reference_words': 0, 'seconds': 0.0} for mode in ('plain','personalized')}
    duration = 0
    silence = {mode: 0 for mode in totals}
    for chunk in rows:
        duration += chunk['duration']
        reference = chunk['corrected'] if chunk['corrected'] is not None else chunk['original']
        for mode in totals:
            start = time.perf_counter()
            prediction = t.transcribe(chunk['path'], chunk['session_id'], chunk['sequence'], personalization=mode == 'personalized')
            elapsed = time.perf_counter() - start
            errors, words = word_errors(reference, prediction)
            totals[mode]['errors'] += errors
            totals[mode]['reference_words'] += words
            totals[mode]['seconds'] += elapsed
            if not words and prediction.strip():
                silence[mode] += 1
    for value in totals.values():
        value['wer'] = value['errors']/value['reference_words'] if value['reference_words'] else None
        value['real_time_factor'] = value['seconds']/max(duration, .001)
    return {'model': model, 'runtime': t.label, 'split': split, 'segments': len(rows),
            'audio_seconds_with_overlap': duration, 'results': totals, 'silence_hallucination_segments': silence,
            'notes': 'No test-label previous context; only train correction evidence. Manual vocabulary and confirmed titles must be frozen before testing. No raw audio or transcripts in this report.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--model', default='small.en')
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--device', choices=['cpu','auto','cuda'], default='cpu')
    p.add_argument('--split', choices=['test','validation'], default='test')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run(Store(args.data), args.model, args.split, args.model_dir, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
