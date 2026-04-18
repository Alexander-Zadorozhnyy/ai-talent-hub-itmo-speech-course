import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class LightningASR(pl.LightningModule):
    def __init__(self, model, vocab, lr: float = 1e-3, blank_token: int = 0):
        super().__init__()
        self.model = model
        self.ctc_loss = nn.CTCLoss(blank=blank_token, zero_infinity=True)
        self.vocab = vocab

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, labels, lengths, label_lenghts = batch

        logits = self(x)
        log_probs = F.log_softmax(logits, dim=-1)

        # input_lengths = lengths // 4

        T_out = logits.size(1)
        input_lengths = torch.full(
            (x.size(0),), T_out, dtype=torch.long, device=x.device
        )

        loss = self.ctc_loss(
            log_probs.permute(1, 0, 2), labels, input_lengths, label_lenghts
        )

        if torch.isnan(loss):
            print("NaN detected!")
            print("logits stats:", logits.mean().item(), logits.std().item())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
