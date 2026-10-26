"""Explicit one-time online setup. Runtime transcription uses only local model files."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


def connections():
    directory = ROOT / 'connections'
    directory.mkdir(exist_ok=True)
    interpreter = str(ROOT / '.venv' / 'Scripts' / 'python.exe')
    script = str(ROOT / 'memory_server.py')
    cfg = {'mcpServers': {'suyash-memory': {'command': interpreter, 'args': [script]}}}
    for name in ('claude-desktop.json', 'gemini-cli.json'):
        (directory / name).write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    toml = ('[mcp_servers.suyash-memory]\ncommand = ' + json.dumps(interpreter) +
            '\nargs = [' + json.dumps(script) + ']\nstartup_timeout_sec = 30\n')
    (directory / 'codex.toml').write_text(toml, encoding='utf-8')
    (directory / 'AI INSTRUCTIONS.txt').write_text(
        'Before continuing a project, use Suyash Shared Memory to search its project name and '
        'read relevant source records. Distinguish raw transcripts, tentative ideas and confirmed '
        'decisions. Resolve conflicting dates or revisions with the user. At the end, propose a '
        'handoff with the goal, completed work, actual artifact paths or links, open questions '
        'and next action. A reference to a file does not grant access to its contents. '
        'If no memory tools are available, ask the user for an exported project handoff.', encoding='utf-8')
    return directory


if __name__ == '__main__':
    import argparse
    from recognition_config import MODEL_IDS
    parser = argparse.ArgumentParser(description='Download a local model without changing the active selection.')
    parser.add_argument('--connections-only', action='store_true')
    parser.add_argument('--model', choices=MODEL_IDS, default='small.en')
    args = parser.parse_args()
    connections()
    if not args.connections_only:
        from faster_whisper.utils import download_model
        target = ROOT / 'models' / args.model
        target.mkdir(parents=True, exist_ok=True)
        print(f'Downloading {args.model}. small.en is about 500 MB; Turbo is about 1.6 GB.')
        download_model(MODEL_IDS[args.model], output_dir=str(target))
        print('Ready. Future transcription runs offline. Open Start Voice.cmd.')
