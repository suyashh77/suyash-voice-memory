"""Merge this server into the standard Claude Desktop config without replacing other settings."""
import json
import os
from datetime import datetime
from pathlib import Path
from memory_store import ROOT


def main():
    config = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming')) / 'Claude' / 'claude_desktop_config.json'
    old = config.read_bytes() if config.exists() else None
    data = json.loads(old.decode('utf-8-sig')) if old else {}
    servers = data.setdefault('mcpServers', {})
    if not isinstance(servers, dict):
        raise ValueError('The existing mcpServers setting is not an object; no changes made.')
    desired = {'command': str(ROOT / '.venv' / 'Scripts' / 'python.exe'),
               'args': [str(ROOT / 'memory_server.py')]}
    if 'suyash-memory' in servers:
        if servers['suyash-memory'] == desired:
            print('Claude Desktop is already configured. Restart it to load the memory tools.')
            return
        raise ValueError('A different suyash-memory server already exists. No changes made.')
    servers['suyash-memory'] = desired
    config.parent.mkdir(parents=True, exist_ok=True)
    if old is not None:
        backup = config.with_name(config.name + '.backup-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
        backup.write_bytes(old)
    if (config.read_bytes() if config.exists() else None) != old:
        raise ValueError('Settings changed while connecting. No changes made; try again.')
    tmp = config.with_suffix('.suyash-memory.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(tmp, config)
    print('Claude Desktop local server configured. Restart Claude Desktop and verify its MCP connection.')


if __name__ == '__main__':
    main()
