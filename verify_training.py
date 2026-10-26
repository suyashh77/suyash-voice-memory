"""Optional training smoke test on a tiny random model and artificial tones, not voice data."""
import argparse
import hashlib
import json
import math
import struct
import uuid
import wave
from pathlib import Path

from memory_store import ROOT
from train_personal_voice import train


def verify(tokenizer_file, folder):
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast
    import torch
    torch.manual_seed(42)
    folder.mkdir(parents=True, exist_ok=False)
    base = folder / 'base'
    tokenizer = WhisperTokenizerFast(tokenizer_file=str(tokenizer_file), language='en', task='transcribe')
    processor = WhisperProcessor(WhisperFeatureExtractor(), tokenizer)
    processor.save_pretrained(str(base))
    config = WhisperConfig(vocab_size=len(tokenizer), num_mel_bins=80, d_model=16,
        encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=1500,
        max_target_positions=448, pad_token_id=50256, bos_token_id=50257,
        eos_token_id=50256, decoder_start_token_id=50257, suppress_tokens=[], begin_suppress_tokens=[])
    WhisperForConditionalGeneration(config).save_pretrained(str(base))
    records = []
    for role, frequency in [('train',220),('validation',440)]:
        pcm = b''.join(struct.pack('<h',int(2000*math.sin(2*math.pi*frequency*i/16000))) for i in range(16000))
        path = folder / (role + '.wav')
        with wave.open(str(path),'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(pcm)
        records.append({'id':role, 'session_id':role, 'audio':path.name, 'text':'synthetic check',
                        'split':role, 'duration':1.0, 'audio_sha256':hashlib.sha256(pcm).hexdigest()})
    manifest = folder / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    args = argparse.Namespace(manifest=manifest, base_model=base, output=folder/'candidate',
        check_only=False,device='cpu',seed=42,epochs=1,learning_rate=1e-4,accumulate=1,convert=True)
    train(args)
    from safetensors.torch import load_file
    weights = load_file(str(args.output/'adapter/adapter_model.safetensors'))
    assert any(float(t.abs().sum())>0 for key,t in weights.items() if 'lora_B' in key)
    from faster_whisper import WhisperModel
    model = WhisperModel(str(args.output/'ct2'),device='cpu',compute_type='int8',cpu_threads=2,local_files_only=True)
    segments,info = model.transcribe(str(folder/'train.wav'),language='en',beam_size=1,max_new_tokens=8,condition_on_previous_text=False)
    list(segments)
    print('PASS: LoRA changed, merged, converted, and CT2 decoded. Artificial labels and tones test the pipeline only.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer',type=Path,default=ROOT/'models/small.en/tokenizer.json')
    p.add_argument('--output',type=Path,default=ROOT/'experiments'/('smoke-'+uuid.uuid4().hex[:10]))
    args = p.parse_args()
    verify(args.tokenizer,args.output)
