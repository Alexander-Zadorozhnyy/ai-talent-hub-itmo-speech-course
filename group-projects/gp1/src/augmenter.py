import torch
import torchaudio
import random
from pathlib import Path
from typing import Optional

class Augmenter:
    def __init__(
        self,
        sample_rate=16000,
        noise_prob=0.5,
        speed_prob=0.5,
        gain_prob=0.3,
        shift_prob=0.5,
        clip_prob=0.2,
        reverb_prob=0.2,
        noise_dir: Optional[Path] = None
    ):
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.noise_paths = (
            list(self.noise_dir.rglob("*.wav")) if self.noise_dir else []
        )
        self.sample_rate = sample_rate

        self.noise_prob = noise_prob
        self.speed_prob = speed_prob
        self.gain_prob = gain_prob
        self.shift_prob = shift_prob
        self.clip_prob = clip_prob
        self.reverb_prob = reverb_prob

    def get_params(self):
        return {
            "noise_prob": self.noise_prob,
            "speed_prob": self.speed_prob,
            "gain_prob": self.gain_prob,
            "shift_prob": self.shift_prob,
            "clip_prob": self.clip_prob,
            "reverb_prob": self.reverb_prob,
        }

    def transform(self, waveform):
        if random.random() < self.shift_prob:
            waveform = self.time_shift(waveform)

        if random.random() < self.speed_prob:
            waveform = self.speed_perturb(waveform)

        if random.random() < self.noise_prob:
            waveform = self.add_noise(waveform)

        if random.random() < self.gain_prob:
            waveform = self.random_gain(waveform)

        if random.random() < self.clip_prob:
            waveform = self.clipping(waveform)

        if random.random() < self.reverb_prob:
            waveform = self.reverb(waveform)

        return waveform

    def add_noise(self, waveform):
        if not self.noise_paths:
            return self._add_gaussian(waveform)
        return self._add_real_noise(waveform)

    def _add_gaussian(self, waveform):
        noise = torch.randn_like(waveform)

        # random SNR between 5–35 dB
        snr_db = random.uniform(5, 35)

        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()

        scale = torch.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power + 1e-6))
        noise = noise * scale

        return waveform + noise
    
    def _add_real_noise(self, waveform):
        noise_path = random.choice(self.noise_paths)
        noise, sr = torchaudio.load(noise_path)
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            noise = torchaudio.functional.resample(noise, sr, self.sample_rate)

        T = waveform.shape[1]
        if noise.shape[1] < T:
            reps = T // noise.shape[1] + 1
            noise = noise.repeat(1, reps)
        start = random.randint(0, noise.shape[1] - T)
        noise = noise[:, start:start + T]

        snr_db = random.uniform(5, 25)
        sig_p = waveform.pow(2).mean()
        noise_p = noise.pow(2).mean()
        scale = torch.sqrt(sig_p / (10 ** (snr_db / 10) * noise_p + 1e-6))
        return waveform + noise * scale

    def speed_perturb(self, waveform):
        speed = random.choice([0.8, 0.9, 1.0, 1.1, 1.2])

        if speed == 1.0:
            return waveform

        new_sr = int(self.sample_rate * speed)
        waveform = torchaudio.functional.resample(waveform, self.sample_rate, new_sr)
        waveform = torchaudio.functional.resample(waveform, new_sr, self.sample_rate)

        return waveform

    def random_gain(self, waveform):
        gain_db = random.uniform(-7, 7)
        return waveform * (10 ** (gain_db / 20))

    def time_shift(self, waveform):
        shift = int(random.uniform(-0.1, 0.1) * waveform.shape[1])
        return torch.roll(waveform, shifts=shift, dims=1)

    def clipping(self, waveform):
        threshold = random.uniform(0.3, 0.8)
        waveform = torch.clamp(waveform, -threshold, threshold)
        return waveform / (threshold + 1e-6)

    def reverb(self, waveform):
        # synthetic impulse response
        ir = torch.zeros(1, 50)
        ir[0, 0] = 1.0
        ir[0, 10] = random.uniform(0.2, 0.5)
        ir[0, 20] = random.uniform(0.1, 0.3)

        return torch.nn.functional.conv1d(
            waveform.unsqueeze(0), ir.unsqueeze(0), padding=25
        ).squeeze(0)
