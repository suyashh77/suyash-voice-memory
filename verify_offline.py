"""End-to-end local pipeline check, using a supplied test WAV (never the microphone)."""
import sys
import tempfile
import time
from memory_store import Store
from speech import Transcriber, import_segments


def main(path):
    with tempfile.TemporaryDirectory() as temp:
        store = Store(temp)
        item = store.create('Validation', 'transcript', 'Synthetic sample')
        count = import_segments(path, store, item['id'])
        transcriber = Transcriber(store)
        transcriber.load()
        start = time.perf_counter()
        for chunk in store.chunks(pending=True):
            text = transcriber.transcribe(chunk['path'])
            store.finish_chunk(chunk['id'], text, 'small.en / CPU int8')
        text = store.get(item['id'])['body']
        assert 'local transcription test' in text.lower(), text
        assert 'share project notes' in text.lower(), text
        assert not store.list(shared_only=True), 'Raw transcripts must stay private'
        print('Full offline pipeline verified: import -> audio segments -> model -> saved private transcript.')
        print('Transcript:', text)
        print('Processing seconds:', round(time.perf_counter() - start, 2))


if __name__ == '__main__':
    main(sys.argv[1])
