# Validation - October 26, 2026

## Runtime and storage

**33 checks passed** across `test_core.py` and `test_adaptation.py`. They cover privacy separation, revisions, concurrent edits, search, handoffs, recoverable audio, correction provenance, long segmented capture, incremental import, learned hint retrieval, reversible evidence, conflicting replacements, project isolation, token budgets, held-out labels, dataset integrity, native-rate neural VAD and lazy CUDA failure fallback.

The real MCP subprocess integration check passed again. All five tools worked, private records were denied, and AI proposals remained outside shared retrieval. The updated recognition-and-learning dialog also passed a desktop construction check. Tests did not open the microphone.

## Offline inference

The complete import -> transcription -> saved private transcript pipeline passed with the updated CPU recognizer. A separate comparison used the same locally synthesized 8.83-second English clip with a 21-word reference:

| Model | Plain decode | Personalized decode | Normalized WER |
| --- | --- | --- | --- |
| small.en, CPU int8, beam 5 | 2.49 s | 1.95 s | 0 for both |
| large-v3-turbo, CPU int8, beam 5 | 8.43 s | 7.48 s | 0 for both |

These are single-clip smoke measurements after loading each model. They include no user-specific training evidence and do not demonstrate a personalization gain. Different warm-up or cache conditions can affect the time. Turbo took substantially longer on this CPU, so small.en remains the default while Turbo is available as an explicit choice.

The held-out benchmark tool reports normalized word errors and processing time. Previous test reference text is excluded from prompts, and only train-role correction evidence supplies learned terms. Manual vocabulary and shared title hints must also be frozen before collecting the test set. Overlapping segments are counted as segments, so their shared audio may receive extra weight in aggregate WER.

## Training scope

Dataset tests verified session-level splitting, exact-duplicate rejection across splits, verified-only labels, and detection of audio modified after export. A tiny random-model check passed through actual LoRA parameter updates, validation-based adapter selection, merging, CTranslate2 conversion and decoding. The optional environment used CPU PyTorch 2.14.1, Transformers 4.57.6, PEFT 0.18.1 and CTranslate2 4.8.4. No personal acoustic model has been trained or shown to outperform its baseline. The reproducible `verify_training.py` uses artificial tones and labels solely to exercise that pipeline.

Real recognition quality for Suyash, noise sensitivity, quiet speech, names, microphone placement, long sessions and meaning-changing errors still need a recorded, verified held-out corpus. Native microphone capture should be checked in everyday use. Successful synthetic tests establish that components run, not that a model is the best recognizer for a particular person.

## Sharing

Recordings, transcripts, datasets, experiments, generated connection paths and model weights remain outside Git. Memory sharing retains the private/proposed/confirmed workflow. Client configuration and client activation remain separate: restart relevant clients and verify their tools. Browser handoffs are snapshots.
