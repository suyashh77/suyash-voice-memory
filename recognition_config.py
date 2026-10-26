"""Explicit model selection. Candidate models never silently replace the active model."""
import json
import os
from pathlib import Path
from memory_store import DATA, ROOT

DEFAULTS = {'model': 'small.en', 'device': 'cpu', 'beam_size': 5, 'language': 'en',
            'personalization': True, 'previous_context': True}
MODEL_IDS = {'small.en': 'Systran/faster-whisper-small.en',
             'large-v3-turbo': 'mobiuslabsgmbh/faster-whisper-large-v3-turbo'}


def load_config(root=DATA):
    path = Path(root) / 'recognition.json'
    config = {**DEFAULTS, **(json.loads(path.read_text(encoding='utf-8')) if path.exists() else {})}
    if config['device'] not in ('cpu', 'cuda', 'auto'):
        raise ValueError('Recognition device must be cpu, cuda or auto')
    if not isinstance(config['beam_size'], int) or not 1 <= config['beam_size'] <= 10:
        raise ValueError('Beam size must be between 1 and 10')
    if (not isinstance(config['model'], str) or not config['model'] or
        config['model'] in ('.','..') or Path(config['model']).name != config['model'] or
        any(c in config['model'] for c in '/\\:')):
        raise ValueError('Select a model folder name inside models/')
    return config


def save_config(config, root=DATA):
    path = Path(root) / 'recognition.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps({**DEFAULTS, **config}, indent=2), encoding='utf-8')
    os.replace(temp, path)


def installed_models():
    return sorted(p.parent.name for p in (ROOT / 'models').glob('*/model.bin'))


def decoder_hints(tokenizer, hints, previous=''):
    """Budget actual model tokens; keep hotwords and natural context separate."""
    encode = lambda text: tokenizer.encode(text).ids
    decode = lambda ids: tokenizer.decode(ids)
    # Both occupy Whisper's previous-text area. Keep the combined budget below 224.
    prompt = 'I am talking through my thoughts and projects in English.'
    if previous:
        prompt += ' ' + decode(encode(previous)[-64:])
    prompt = decode(encode(prompt)[:90])
    chosen = []
    for phrase in hints:
        candidate = ', '.join(chosen + [phrase])
        if len(encode(' ' + candidate)) <= 90:
            chosen.append(phrase)
    return prompt, ', '.join(chosen) or None
