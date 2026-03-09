import pytest
import torch
import torchaudio

from melbanks import LogMelFilterBanks


@pytest.fixture
def audio_sample():
    """Fixture to load the audio sample."""
    signal, sr = torchaudio.load("./examples/ff-16b-1c-16000hz.mp3")
    return signal, sr


@pytest.mark.parametrize(
    "hop_length, n_mels, mel_scale, power",
    [
        (160, 80, "htk", 2),
        (320, 40, "htk", 2),
        (80, 160, "htk", 2),
        (160, 80, "slaney", 2),
        (160, 80, "slaney", 4),
    ],
)
def test_different_parameters(audio_sample, hop_length, n_mels, mel_scale, power):
    """Test with different MelSpectrogram parameters."""
    signal, _ = audio_sample

    melspec = torchaudio.transforms.MelSpectrogram(
        hop_length=hop_length,
        n_mels=n_mels,
        mel_scale=mel_scale,
        power=power,
    )(signal)
    logmelbanks = LogMelFilterBanks(
        hop_length=hop_length,
        n_mels=n_mels,
        mel_scale=mel_scale,
        power=power,
    )(signal)

    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)
