# Personal transcription audit and changes

## Situation

The original application reliably collected local speech, corrections and project memory, but its transcriber never read the corrections table. Manual vocabulary hints were the only feedback that affected future decoding. A source audit confirmed that gap.

The starting source was commit `8dba576`. Its actual capture thresholds were ten seconds minimum, 0.7 seconds of low RMS, and 25 seconds maximum. Microphone audio used the device's native sample rate, not a fixed 16 kHz. The pasted review's four-second and RMS-350 details did not match this version.

## Task

Make correction feedback useful immediately, improve pause handling and model selection, and create a measurable route to acoustic adaptation. Keep recordings local and preserve the distinction between raw speech, user corrections and shared memory.

## Analysis

Personalization is a testable hypothesis. Repeated pronunciations, names and vocabulary can provide useful evidence, but a person's speech varies with fatigue, microphone position, noise, topic and speaking style. A personal system may outperform a baseline on a defined test set. There is no basis here for promising that it will beat every model for every future recording.

The assertion that 80% of personal errors are lexical is unsupported by this project's data. A word error alone does not identify whether the encoder, language prior, segmentation, microphone or human label caused it. We should examine error categories before choosing what to train.

The CTranslate2 point is sound: it is an inference engine, so training requires a separate trainable model and framework. Per-edit online learning is technically possible in other systems, but isolated, unvalidated updates can overfit or regress. This application therefore uses immediate lexical learning and leaves weight adaptation as a separate batch experiment. [CTranslate2](https://github.com/OpenNMT/CTranslate2), [PEFT ASR guide](https://huggingface.co/docs/peft/main/en/task_guides/int8-asr).

Turbo reduces Whisper large-v3's decoder from 32 layers to four. Its model card explicitly describes a small quality degradation in return for speed. It does not establish equal accuracy or near-small speed on this laptop. GPU presence also does not guarantee working CUDA libraries or enough free VRAM. [Original model card](https://huggingface.co/openai/whisper-large-v3-turbo).

In the installed faster-whisper implementation, hotwords and previous text both enter the decoder prompt. Moving a word list into `hotwords` alone does not turn it into a hard vocabulary constraint or solve prompt limits. The upgrade budgets both inputs with the actual tokenizer and uses natural prose plus bounded prior-session context. `condition_on_previous_text` only concerns internal windows of one transcribe call. Independent app chunks need explicit context even if that flag is enabled. [faster-whisper implementation](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py).

Neural VAD improves the evidence used to find pauses, but it cannot create a pause in uninterrupted speech. Maximum-duration boundaries still need a fallback. The implementation retains overlap there and preserves the original audio before detection. It uses the bundled Silero ONNX model and its recurrent state, without introducing PyTorch into live recording. [Silero VAD](https://github.com/snakers4/silero-vad).

LoRA does not guarantee protection from forgetting. Thirty or sixty minutes can be a useful experimental milestone, and rehearsal can help, but neither is a universal readiness threshold. Selection must use separate validation audio and a regression set. Training time is hardware-dependent. The optional trainer produces a candidate and never automatically activates it. [PEFT ASR](https://huggingface.co/docs/peft/main/en/task_guides/int8-asr), [Whisper conversion](https://opennmt.net/CTranslate2/guides/transformers.html).

A free-form LLM postprocessor was deliberately omitted. It could make an inaccurate transcript read fluently while changing intent. Approved, inspectable phrase rules provide a narrower first experiment. Further postprocessing should be evaluated separately against the same held-out references.

## Result

The upgrade connects correction feedback to project-specific hint retrieval. Learned phrase substitutions require explicit approval, retain original recognition, and stop applying when their evidence disappears or conflicts. Multiple saves of one audio segment do not masquerade as independent examples.

Model selection, token-budgeted context, neural pause detection, verified dataset export, held-out WER comparison, and an optional offline LoRA/CT2 candidate script are included. Personal recordings, labels, datasets and trained weights remain excluded from Git.

The next evidence needed is a set of natural recordings with exact transcripts in separate sessions. Compare small.en and Turbo, with and without lexical adaptation. Track important-name errors, meaning-changing errors and processing delay alongside WER. Only then decide whether extra acoustic training earns its cost in setup time and compute.
