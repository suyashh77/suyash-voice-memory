"""Optional offline LoRA experiment. Never runs inside recording or activates its output."""
import argparse
import hashlib
import json
from pathlib import Path


def read_manifest(path):
    manifest = Path(path).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    groups, hashes = {}, {}
    for row in rows:
        if row['split'] not in ('train','validation','test'):
            raise ValueError('Unknown dataset split')
        groups.setdefault(row['session_id'], set()).add(row['split'])
        hashes.setdefault(row['audio_sha256'], set()).add(row['split'])
        path = (manifest.parent / row['audio']).resolve()
        if not path.is_relative_to(manifest.parent) or not path.is_file():
            raise ValueError('Dataset audio must be inside the exported dataset')
        if not row['text'].strip() or not 0 < row['duration'] <= 30:
            raise ValueError('Invalid verified label or duration')
        import wave
        with wave.open(str(path), 'rb') as audio:
            if audio.getframerate() != 16000 or audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError('Training audio must be mono 16 kHz PCM16')
            pcm = audio.readframes(audio.getnframes())
        if hashlib.sha256(pcm).hexdigest() != row['audio_sha256']:
            raise ValueError('Dataset audio changed after export')
        row['path'] = str(path)
    if any(len(v) != 1 for v in (*groups.values(), *hashes.values())):
        raise ValueError('Session or duplicate audio crosses splits')
    if not any(r['split'] == 'train' for r in rows) or not any(r['split'] == 'validation' for r in rows):
        raise ValueError('Training needs separate train and validation sessions. Keep test sessions untouched.')
    return rows


def train(args):
    rows = read_manifest(args.manifest)
    summary = {role: {'segments': sum(r['split'] == role for r in rows),
                      'minutes': sum(r['duration'] for r in rows if r['split'] == role)/60}
               for role in ('train','validation','test')}
    print(json.dumps(summary, indent=2))
    if args.check_only:
        return summary
    if not args.base_model.is_dir():
        raise ValueError('Provide a local Hugging Face Whisper checkpoint. CT2 weights cannot train.')
    if args.output.exists():
        raise ValueError('Use a fresh output folder; do not overwrite a previous experiment.')
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast
    from peft import LoraConfig, get_peft_model
    from faster_whisper.audio import decode_audio
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    processor = WhisperProcessor.from_pretrained(str(args.base_model), local_files_only=True)
    processor.tokenizer = WhisperTokenizerFast.from_pretrained(str(args.base_model), local_files_only=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if processor.tokenizer.pad_token_id is None:
        raise ValueError('Whisper tokenizer needs an end-of-text/padding token')
    model = WhisperForConditionalGeneration.from_pretrained(str(args.base_model), local_files_only=True)
    processor.tokenizer.set_prefix_tokens(language='en', task='transcribe')
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                           target_modules=['q_proj','v_proj'], bias='none'))
    model.config.use_cache = False
    device = 'cuda' if args.device == 'cuda' else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA training requested but unavailable. CPU training is much slower.')
    model.to(device)
    class Examples(Dataset):
        def __init__(self, role):
            self.rows = [r for r in rows if r['split'] == role]
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            audio = decode_audio(row['path'], sampling_rate=16000)
            feature = processor.feature_extractor(audio, sampling_rate=16000, return_tensors='pt').input_features[0]
            label = processor.tokenizer(row['text']).input_ids
            if len(label) > 440:
                raise ValueError('Label too long for Whisper. Check the transcript/audio alignment.')
            return feature, label
    def collate(items):
        features, labels = zip(*items)
        packed = processor.tokenizer.pad([{'input_ids': label} for label in labels], return_tensors='pt')
        values = packed.input_ids.masked_fill(packed.attention_mask.ne(1), -100)
        if (values[:, 0] == model.config.decoder_start_token_id).all():
            values = values[:, 1:]
        return {'input_features': torch.stack(features).to(device), 'labels': values.to(device)}
    train_loader = DataLoader(Examples('train'), batch_size=1, shuffle=True, collate_fn=collate)
    validation = DataLoader(Examples('validation'), batch_size=1, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    args.output.mkdir(parents=True)
    history, best = [], float('inf')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            count = min(args.accumulate, len(train_loader) - (step//args.accumulate)*args.accumulate)
            loss = model(**batch).loss / count
            loss.backward()
            if (step+1) % args.accumulate == 0 or step+1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
                optimizer.step()
                optimizer.zero_grad()
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in validation:
                losses.append(float(model(**batch).loss))
        score = sum(losses)/len(losses)
        history.append({'epoch': epoch+1, 'validation_loss': score})
        if score < best:
            best = score
            model.save_pretrained(str(args.output / 'adapter'))
        print(history[-1], flush=True)
    # Restore the best validation-loss adapter before merging. WER is evaluated separately.
    del optimizer, train_loader, validation, model
    import gc
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    from peft import PeftModel
    base = WhisperForConditionalGeneration.from_pretrained(str(args.base_model), local_files_only=True)
    best_model = PeftModel.from_pretrained(base, str(args.output / 'adapter'))
    merged = best_model.merge_and_unload()
    merged.config.use_cache = True
    merged_path = args.output / 'merged'
    merged.save_pretrained(str(merged_path), safe_serialization=True)
    processor.save_pretrained(str(merged_path))
    metadata = {'base_model': str(args.base_model), 'epochs': history, 'seed': args.seed,
                'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'summary': summary, 'training_sessions': sorted({r['session_id'] for r in rows if r['split']=='train'}),
                'status': 'candidate; evaluate held-out WER and latency before explicit activation'}
    (args.output / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if args.convert:
        from ctranslate2.converters import TransformersConverter
        TransformersConverter(str(merged_path), copy_files=['tokenizer.json','preprocessor_config.json']).convert(
            str(args.output / 'ct2'), quantization='int8', force=False)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--convert', action='store_true')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20 or args.accumulate < 1 or args.learning_rate <= 0:
        parser.error('Use 1–20 epochs, positive accumulation and a positive learning rate')
    train(args)
