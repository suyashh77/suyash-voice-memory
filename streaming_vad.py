"""Stateful CPU Silero VAD using the ONNX asset bundled with faster-whisper 1.2.1."""
import numpy as np


class StreamingVAD:
    def __init__(self, rate):
        import av
        from faster_whisper.vad import get_vad_model
        self.rate = rate
        self.session = get_vad_model().session
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros_like(self.h)
        self.context = np.zeros(64, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        self.quiet_seconds = 0.0
        self.speaking = False

    def feed(self, pcm):
        import av
        audio = np.frombuffer(pcm, dtype='<i2').reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(audio, format='s16', layout='mono')
        frame.sample_rate = self.rate
        for converted in self.resampler.resample(frame):
            self.pending = np.concatenate((self.pending, converted.to_ndarray().ravel().astype(np.float32)/32768))
        offset = 0
        while len(self.pending) - offset >= 512:
            samples = self.pending[offset:offset+512]
            inputs = np.concatenate((self.context, samples)).reshape(1, -1)
            out, self.h, self.c = self.session.run(None, {'input': inputs, 'h': self.h, 'c': self.c})
            probability = float(np.asarray(out).ravel()[0])
            self.context = samples[-64:].copy()
            if probability >= .5:
                self.speaking = True
                self.quiet_seconds = 0
            elif probability < .35:
                self.quiet_seconds += 512 / 16000
                if self.quiet_seconds >= .7:
                    self.speaking = False
            else:
                # Ambiguous frames are not evidence of a continuous pause.
                self.quiet_seconds = 0
            offset += 512
        self.pending = self.pending[offset:]
        return self.quiet_seconds
