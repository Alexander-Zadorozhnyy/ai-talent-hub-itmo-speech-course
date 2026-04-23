import torch 
import torch.nn as nn 
import torchaudio

class ASRModel(nn.Module):
    """Conformer encoder. Input mel (B, T, n_mels), output logits (B, T', vocab)"""
    downsample_factor = 4

    def __init__(self, vocab_size, n_mels=80, d_model=176, num_layers=6, num_heads=4, 
                 ffn_dim=256, conv_kernel_size=31, dropout=0.1, subsample_channels=16):
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv2d(1, subsample_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(subsample_channels, subsample_channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(subsample_channels * (n_mels // 4), d_model)
        self.conformer = torchaudio.models.Conformer(
            input_dim=d_model,
            num_heads=num_heads, 
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        x = x.unsqueeze(1)
        x = self.subsample(x)
        B, C, Tp, Fp = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, Tp, C * Fp)
        x = self.proj(x)

        if lengths is None:
            lengths = torch.full((B,), Tp, dtype=torch.long, device=x.device)
        else:
            lengths = (((lengths + 1) // 2 + 1) // 2).clamp(max=Tp).to(x.device)
        x, _ = self.conformer(x, lengths)
        return self.fc(x)