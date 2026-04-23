import torch.nn as nn


class ASRModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.hidden_size = 256

        self.lstm = nn.LSTM(
            input_size=64 * 20,  # depends on mel dims
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(self.hidden_size * 2, vocab_size)

    def forward(self, x, lengths=None):
        # x: (B, T)
        x = x.unsqueeze(1)  # (B, 1, T)

        x = self.cnn(x)

        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)

        x, _ = self.lstm(x)

        x = self.fc(x)

        return x
