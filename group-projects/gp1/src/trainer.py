import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict
from src.metrics import decode_batch, cer, harmonic_cer



class LightningASR(pl.LightningModule):
    def __init__(self, model, vocab, normalizer, in_domain_speakers, lr: float = 1e-3, blank_token: int = 0):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "vocab", "normalizer", "in_domain_speakers"]) 
        self.model = model
        self.ctc_loss = nn.CTCLoss(blank=blank_token, zero_infinity=True)
        self.vocab = vocab
        self.normalizer = normalizer
        self.in_domain_speakers = set(in_domain_speakers)
        self.lr = lr
        self.val_outputs = [] 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, labels, lengths, label_lengths, _ = batch

        logits = self(x)
        log_probs = F.log_softmax(logits, dim=-1)

        T_out = logits.size(1)
        input_lengths = ((lengths + 1) // 2 + 1) // 2
        input_lengths = input_lengths.clamp(max=T_out).to(x.device)

        assert (input_lengths >= label_lengths).all()

        loss = self.ctc_loss(
            log_probs.permute(1, 0, 2), labels, input_lengths, label_lengths
        )

        if torch.isnan(loss):
            print("NaN detected!")
            print("logits stats:", logits.mean().item(), logits.std().item())

        self.log("train_loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):

        x, labels, lengths, label_lengths, spk_ids = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=-1)

        for (pred_text, gt_text), spk in zip(
            decode_batch(preds, labels, label_lengths, self.normalizer),
            spk_ids,
        ):
            self.val_outputs.append((cer(pred_text, gt_text), spk))
    def on_validation_epoch_end(self):
        per_spk = defaultdict(list)
        in_d, ood = [], [] 
        for c, spk in self.val_outputs:
            per_spk[spk].append(c)
            (in_d if spk in self.in_domain_speakers else ood).append(c)
        
        in_d_mean = sum(in_d) / len(in_d) if in_d else 0.0 
        ood_mean = sum(ood) / len(ood) if ood else 0.0

        self.log("val/cer_in_domain", in_d_mean, prog_bar=True)
        self.log("val/cer_out_of_domain", ood_mean, prog_bar=True)
        self.log("val/harmonic_cer", harmonic_cer(in_d_mean, ood_mean), prog_bar=True)
        for spk, vals in per_spk.items():
            self.log(f"val/cer_{spk}", sum(vals) / len(vals))
        
        self.val_outputs.clear()

            



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
