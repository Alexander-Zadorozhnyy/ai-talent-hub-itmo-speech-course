from pathlib import Path
import torch
import torchaudio
from src.normalizers.unspittable_normalizer import UnspittableNormalizer
from src.models.conformer import ASRModel
from src.metrics import ctc_decode

class ASRInference:
    def __init__(self, ckpt_path: Path, device: str = "cuda"):
        self.device = device
        self.normalizer = UnspittableNormalizer()
        _, vocab_size = self.normalizer.get_vocab()
        self.model = ASRModel(vocab_size).to(device)

        state = torch.load(ckpt_path, map_location=device)
        # Lightning saves under "state_dict" with "model." prefix from LightningASR
        model_state = {k.replace("model.", "", 1): v
                       for k, v in state["state_dict"].items()
                       if k.startswith("model.")}
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=256, hop_length=160, n_mels=80
        ).to(device)

    def _load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.to(self.device)

    def _to_mel(self, wav):
        m = self.mel(wav)
        m = torch.log(m + 1e-6)
        m = (m - m.mean()) / (m.std() + 1e-6)
        return m.squeeze(0).transpose(0, 1)   # (T, n_mels)

    @torch.no_grad()
    def transcribe(self, paths: list[Path]) -> list[str]:
        results = []
        for p in paths:
            mel = self._to_mel(self._load_audio(p)).unsqueeze(0)   # (1, T, 80)
            logits = self.model(mel)
            pred = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            tokens = ctc_decode(pred)
            results.append(self.normalizer.tokens2num(tokens))
        return results
