from typing import Optional

import torch
import torchaudio
import pandas as pd
from tqdm import tqdm

from src.normalizers.normalizer import Normalizer
from src.augmenter import Augmenter


class ASRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path,
        root_dir,
        normalizer: Normalizer,
        augmenter: Optional[Augmenter] = None,
        sample_rate: int = 16000,
        n_fft: int = 256,
        hop_length: int = 160,
        n_mels: int = 80,
        device: torch.device = torch.device("cpu"),
    ):
        self.normalizer = normalizer
        self.augmenter = augmenter

        self.root_dir = root_dir
        self.device = device

        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        self.df = pd.read_csv(csv_path)

        self.audio_paths = self.df["filename"].tolist()
        self.audio_files = [
            self.load_audio(self.root_dir / path)
            for path in tqdm(self.audio_paths, desc=f"Loading audio files")
        ]

        self.labels = self.df["transcription"].tolist()
        self.normalized_labels = [
            torch.tensor(self.normalizer.num2tokens(label)).to(self.device)
            for label in tqdm(self.labels, desc=f"Generating label files")
        ]

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)

        # Convert to mono audio
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to the specified sample rate
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        return waveform

    def __getitem__(self, idx):
        audio = self.audio_files[idx]

        if self.augmenter is not None:
            audio = self.augmenter.transform(audio)

        label = self.normalized_labels[idx]
        return self.to_mel(audio), label

    def to_mel(self, audio):
        mel_spec = self.mel_transform(audio)
        mel_spec = torch.log(mel_spec + 1e-6)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        return mel_spec.squeeze(0).transpose(0, 1).to(self.device)
