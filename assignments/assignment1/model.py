import time

import torch
import torch.nn as nn
import pytorch_lightning as pl
from thop import profile
from melbanks import LogMelFilterBanks


class SampleCNNModel(nn.Module):
    def __init__(self, n_mels: int = 80, n_classes: int = 2, groups: int = 1):
        super().__init__()

        self.groups = groups
        self.base_channel = 32

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=n_mels,
                out_channels=self.base_channel,
                kernel_size=3,
                padding=1,
                groups=groups,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=self.base_channel,
                out_channels=self.base_channel * 2,
                kernel_size=3,
                padding=1,
                groups=groups,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=self.base_channel * 2,
                out_channels=self.base_channel * 4,
                kernel_size=5,
                padding=1,
                groups=groups,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(self.base_channel * 4, n_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class BinarySpeechClassifier(pl.LightningModule):
    def __init__(self, n_mels: int = 80, lr: float = 1e-3, groups: int = 1):
        super().__init__()

        self.feature_extractor = LogMelFilterBanks(n_mels=n_mels)
        self.model = SampleCNNModel(n_mels=n_mels, n_classes=2, groups=groups)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

        self.start_time = None
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.model(features)
        return logits

    def step(self, batch, calc_loss: bool = True):
        waves, labels = batch
        logits = self.forward(waves)
        loss = self.criterion(logits, labels) if calc_loss else None

        return labels, logits, loss

    def on_train_epoch_start(self):
        self.start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.start_time
        self.log("epoch_time", epoch_time)

    def training_step(self, batch, batch_idx):
        _, _, loss = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels, logits, loss = self.step(batch)
        preds = torch.argmax(logits, dim=1)

        acc = (preds == labels).float().mean()
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_accuracy", acc, prog_bar=True)
        return acc

    def test_step(self, batch, batch_idx):
        labels, logits, _ = self.step(batch, calc_loss=False)
        preds = torch.argmax(logits, dim=1)

        acc = (preds == labels).float().mean()
        self.log("test_accuracy", acc)
        self.compute_flops()
        return acc

    def compute_flops(self, input_size=(1, 16000)):
        dummy_input = torch.randn(*input_size).to(self.device)
        features = self.feature_extractor(dummy_input)
        
        macs, params = profile(self.model, inputs=(features,))
        flops = macs * 2

        print("Model FLOPs: %s Params: %s \n" % (flops, params))
        self.log_dict({"flops": flops, "params": params})
