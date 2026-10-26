"""Integration check through a real MCP subprocess, using temporary data only."""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anyio
from memory_store import ROOT, Store


async def main():
    with tempfile.TemporaryDirectory() as temp:
        store = Store(temp)
        shared = store.create('Voice', 'decision', 'Local speech', 'Use offline transcription.', source='User', status='confirmed')
        private = store.create('Hidden', 'note', 'Do not share', 'secret kiwi')
        params = StdioServerParameters(command=sys.executable,
                    args=[str(ROOT / 'memory_server.py')],
                    env={**os.environ, 'SUYASH_VOICE_DATA': temp})
        print('Starting MCP subprocess…', flush=True)
        async with stdio_client(params) as (reader, writer):
            print('Subprocess started; initializing…', flush=True)
            async with ClientSession(reader, writer) as session:
                with anyio.fail_after(20):
                    initialized = await session.initialize()
                print('Initialized; checking tools…', flush=True)
                tools = await session.list_tools()
                assert len(tools.tools) == 5
                found = await session.call_tool('search_memories', {'query': 'offline', 'project': 'Voice'})
                assert not found.isError and shared['id'] in str(found)
                hidden = await session.call_tool('read_memory', {'memory_id': private['id']})
                assert hidden.isError
                proposed = await session.call_tool('propose_memory', {'project': 'Voice', 'title': 'Next action',
                    'body': 'Test Suyash voice accuracy.', 'source': 'Integration test'})
                assert not proposed.isError
                assert not store.list(query='accuracy', shared_only=True)
                contexts = await session.call_tool('project_context', {'project': 'Voice'})
                assert not contexts.isError and 'secret kiwi' not in str(contexts)
                memory_map = await session.call_tool('memory_map', {})
                assert 'Hidden' not in str(memory_map)
                print('MCP verified: initialize, 5 tools, search, context, map, private denial, proposal isolation.')


if __name__ == '__main__':
    asyncio.run(main())
