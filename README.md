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

The app preserves original audio, original recognition, adjusted text, approved replacements, and corrected text. Saving a correction now mines short phrase substitutions and adds likely names or recurring terms to project-specific decoder hints. Re-speaking uses the same recognizer, so its proposed correction can also be wrong.

Open **Recognition & learning** to review learned replacements. Explicit approval enables whole-word substitutions for that project. Conflicting examples disable a rule, and saving the same segment repeatedly counts as one example. Reverting a label removes its learned evidence. Wholesale rewrites do not create phrase rules.

Prompt context and hotwords have separate budgets measured with the active model's tokenizer. Only the current session's preceding segments supply speech context. Confirmed memory titles can supply names, while memory bodies stay out of the decoder prompt. This limits prompt pollution and avoids turning long notes into invented speech.

Live capture uses stateful Silero neural VAD on the writer thread. Audio is saved before pause detection. Natural pauses split segments after at least three seconds. Continuous speech still needs a maximum-duration boundary, so the 25-second fallback retains a short overlap. If VAD fails, fixed-duration recording continues with a visible status message. Saved microphone WAVs retain the microphone's native rate. Dataset exports resample to mono 16 kHz.

## Model selection and evaluation

**Install Turbo.cmd** downloads `large-v3-turbo` using the model mapping supported by faster-whisper. Choose an installed model in **Recognition & learning**, then restart. `small.en` remains the default until you choose otherwise. CPU uses int8 and CUDA uses int8_float16, with CPU retry if model loading or actual GPU decoding fails. CUDA libraries and a healthy driver are still required for GPU operation. Both CPU models use beam size 5 by default.

Turbo is a candidate for better recognition, not a guaranteed winner for every speaker. Its original model card describes a quality/speed tradeoff relative to large-v3. Compare accuracy and processing time on your actual recordings before choosing a default.

To measure improvement, reserve separate recording sessions as **validation** or **test** in Recognition & learning, then listen and verify their words. The split applies to the whole session, including overlapping neighbors and later segments. These labels do not teach hints or replacements. Freeze manually added vocabulary and shared title hints before recording test sessions too.

```powershell
python benchmark_voice.py --data data --model small.en --output benchmarks/small.json
python benchmark_voice.py --data data --model large-v3-turbo --output benchmarks/turbo.json
```

Each report compares plain and personalized decoding on the same verified test segments and reports word error rate and processing time. It excludes previous reference transcripts from decoder context and contains aggregate scores only. Warm-up, sample size, microphone conditions, and overlap affect interpretation. A lower score on a few recordings cannot establish superiority to every commercial or open model.

## Optional offline weight adaptation

The recording app never runs backpropagation. Its CTranslate2 backend performs inference. An optional research path exports verified labels, trains LoRA adapters with PyTorch, and converts a merged candidate back to CTranslate2:

```powershell
python personal_dataset.py datasets/experiment-001 --data data
python train_personal_voice.py datasets/experiment-001/manifest.jsonl --base-model LOCAL_HF_WHISPER_FOLDER --output experiments/run-001 --check-only
```

The exporter requires a fresh destination and exports only user-verified chunks. It retains source IDs, groups by session, detects exact duplicate audio across splits, and creates mono 16 kHz WAV files. For training, use a separate environment with `requirements-training.txt` and a locally downloaded Hugging Face Whisper checkpoint matching the intended base model. CTranslate2 inference weights cannot be used as trainable PyTorch weights.

Remove `--check-only` to train, and add `--convert` to create `experiments/run-001/ct2`. The optional trainer applies rank-16 LoRA to query/value attention projections, uses separate validation sessions to select an adapter by validation loss, and leaves test sessions untouched. Before activation, compare candidate WER and latency against the base model on held-out audio. Candidate conversion and activation are separate steps. Copy an accepted CT2 candidate into a new `models/` subfolder and select it explicitly.

This training path passed a tiny random-model smoke test through LoRA updates, adapter selection, merging, CTranslate2 conversion and local decoding. That validates the pipeline only. No personal acoustic model has been trained or demonstrated an accuracy gain. In the optional training environment, `python verify_training.py` repeats a small pipeline check using artificial tones and labels, not voice data. It requires the installed small.en tokenizer file.

Training time and memory depend on model and hardware. No fixed amount of corrected speech or rehearsal percentage guarantees improvement. Include verified unchanged examples and varied recording conditions, keep a general-speech regression set, and increase training only when measured results support it. No automatic training schedule or cloud upload is configured.

## Shared memory

The **Shared memory** tab stores notes, preferences, decisions, handoffs, and raw transcripts by project. SQLite is the authoritative store. Markdown files are readable exports, and changes made directly to an export do not update the database.

Records have three sharing states:

- **Private:** visible in the local app only.
- **Proposed:** submitted for review, including suggestions from an AI.
- **Confirmed:** available to connected AI tools and included in project handoffs.

Recordings create private transcripts. **Share as a note** creates a shared snapshot of a transcript. Check the words and meaning before relying on that snapshot. Approval to share a transcript does not turn tentative speech into a verified fact.

The MCP server exposes five tools: `memory_map`, `search_memories`, `read_memory`, `project_context`, and `propose_memory`. Retrieval includes source information and revisions. AI proposals stay out of retrieval until the user confirms them in the app. Revision checks prevent stale edits from silently replacing newer records.

Memory search uses SQLite full-text keyword matching. Semantic memory embeddings, a knowledge graph, automatic chat-history import, voice cloning, and video generation remain outside this version. The optional ASR adaptation tools operate locally and do not change memory sharing states.

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
python -m unittest test_core test_adaptation -v
python verify_mcp.py
python verify_offline.py path/to/synthetic-test.wav
```

`test_core.py` checks storage privacy, concurrent updates, revisions, search, handoffs, correction provenance, recoverable audio, segmented capture, incremental import, and desktop widget construction. It does not activate the microphone. `verify_mcp.py` exercises a real local server using temporary data.

`verify_offline.py` expects test speech containing "local transcription test" and "share project notes". Model files must be installed first. This is a pipeline check, not an accuracy benchmark. See [VALIDATION.md](VALIDATION.md) for the original test scope and remaining limits.

Tests that construct Tkinter widgets need a desktop session. Windows sandbox restrictions on local loopback can prevent the asynchronous MCP test from starting.
