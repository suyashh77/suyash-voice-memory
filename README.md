# Suyash Voice & Memory

A Windows desktop app for local speech-to-text and a shared memory library that AI clients can search. It uses faster-whisper for transcription, SQLite for storage, Markdown for portable handoffs, and a local MCP server for connected clients.

Recordings stay on the computer. There is no transcription API key or per-minute provider fee. Initial setup downloads Python dependencies and an English speech model, and normal use still depends on available disk space and computer performance.

## Run it

Install Python 3.12 on Windows, clone this repository, and double-click **Setup Voice.cmd** while online. Setup creates an isolated Python environment, downloads the Whisper `small.en` model, and generates connection configurations with paths for this installation.

Double-click **Start Voice.cmd** after setup. Wait for the local model to load, choose a microphone, and click **Start talking**. Click **Stop recording** when finished. Ctrl + Space starts or stops capture while the app window is focused.

Text appears after a pause or a short audio segment, followed by processing time. This version transcribes segments rather than displaying every word instantly. Use **Copy text** to paste into a chat or notes app. **Import audio** handles existing files incrementally.

The runtime loads the downloaded model locally and does not send recordings to a transcription service. The default configuration uses CPU int8 inference. GPU acceleration and multilingual models need separate configuration and testing.

## Corrections

Add names and short phrases in **My vocabulary** to guide future recognition. For feedback, select a segment, listen to it, then edit the text or use **Re-say correction**. Stop the correction recording, review its words, and choose **Save correction**.

The app preserves original audio, original text, and corrected text. These pairs could support future model adaptation, but this version does not retrain itself. Re-speaking uses the same recognizer, so its proposed correction can also be wrong.

## Shared memory

The **Shared memory** tab stores notes, preferences, decisions, handoffs, and raw transcripts by project. SQLite is the authoritative store. Markdown files are readable exports, and changes made directly to an export do not update the database.

Records have three sharing states:

- **Private:** visible in the local app only.
- **Proposed:** submitted for review, including suggestions from an AI.
- **Confirmed:** available to connected AI tools and included in project handoffs.

Recordings create private transcripts. **Share as a note** creates a shared snapshot of a transcript. Check the words and meaning before relying on that snapshot. Approval to share a transcript does not turn tentative speech into a verified fact.

The MCP server exposes five tools: `memory_map`, `search_memories`, `read_memory`, `project_context`, and `propose_memory`. Retrieval includes source information and revisions. AI proposals stay out of retrieval until the user confirms them in the app. Revision checks prevent stale edits from silently replacing newer records.

Search uses SQLite full-text keyword matching. Semantic embeddings, a knowledge graph, automatic chat-history import, acoustic fine-tuning, voice cloning, and video generation are outside this version.

## Connect AI clients

Setup generates these local files in `connections/`:

| File | Purpose |
| --- | --- |
| `claude-desktop.json` | Local MCP server configuration for Claude Desktop |
| `codex.toml` | Local MCP server entry for Codex-compatible clients |
| `gemini-cli.json` | Local MCP server configuration for Gemini CLI |
| `AI INSTRUCTIONS.txt` | Suggested guidance for searching memory and leaving handoffs |

Merge the generated server entry into the client's settings, preserving other settings. **Connect Claude.cmd** can add this server to the standard Claude Desktop configuration and back up an existing file. Restart the client and verify that its tools appear. Configuration alone does not establish an active connection.

References: [Claude Desktop local servers](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop), [OpenAI MCP connections](https://learn.chatgpt.com/docs/extend/mcp?surface=cli), [Gemini CLI MCP servers](https://geminicli.com/docs/tools/mcp-server/). Client capabilities and account policies can vary.

Browser chats can use **Copy project handoff** to receive a context file by paste or upload. This portable workflow works with ChatGPT, Gemini, Muse, and other apps that accept text. It is a snapshot, not automatic synchronization. This repository does not publish a remote memory server or include a Muse-specific connector.

A practical handoff contains the project goal, decisions, completed work, actual artifact paths or links, open questions, and the next action. The receiving AI also needs access to the actual files. A memory containing a filename does not provide its contents.

## Data and recovery

Runtime files live in `data/`: the SQLite database, audio pieces, Markdown memory exports, handoff exports, and an error log when needed. Transcription can continue while new audio is recorded. Pending audio stays on disk and resumes when the app starts again. Startup also checks for recoverable unindexed WAV pieces.

There is no configured duration quota or automatic retention policy. Disk space still limits recording. WAV audio uses about 115 MB per hour at 16 kHz or 346 MB per hour at 48 kHz. Operating-system buffering and power loss can affect recovery, so keep backups. For a simple folder backup, close the app and connected memory processes first.

Data is ordinary local storage and is not encrypted by this app. When a cloud AI retrieves or receives shared memory, that context goes to its provider. Private data, recordings, downloaded models, generated connection paths, and the Python environment are excluded from this repository.

## Development and validation

The application uses Python, Tkinter, sounddevice, faster-whisper, and the official Python MCP SDK. `requirements.txt` pins direct dependencies, including a compatible PyAV version. `requirements-lock.txt` records the dependencies used in the original Windows validation.

After installing dependencies, run:

```powershell
python -m unittest test_core -v
python verify_mcp.py
python verify_offline.py path/to/synthetic-test.wav
```

`test_core.py` checks storage privacy, concurrent updates, revisions, search, handoffs, correction provenance, recoverable audio, segmented capture, incremental import, and desktop widget construction. It does not activate the microphone. `verify_mcp.py` exercises a real local server using temporary data.

`verify_offline.py` expects test speech containing "local transcription test" and "share project notes". Model files must be installed first. This is a pipeline check, not an accuracy benchmark. See [VALIDATION.md](VALIDATION.md) for the original test scope and remaining limits.

Tests that construct Tkinter widgets need a desktop session. Windows sandbox restrictions on local loopback can prevent the asynchronous MCP test from starting.
