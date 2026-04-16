from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from model import BinarySpeechClassifier
from dataset import BinarySpeechCommands


def main():
    train_loader, test_loader, val_loader = get_dataset_loaders(num_workers=5)
    epochs = 20

    params = [
        {"n_mels": 20, "lr": 1e-3, "groups": 1, "epochs": epochs},
        {"n_mels": 40, "lr": 1e-3, "groups": 1, "epochs": epochs},
        {"n_mels": 80, "lr": 1e-3, "groups": 1, "epochs": epochs},
        {"n_mels": 40, "lr": 1e-3, "groups": 2, "epochs": epochs},
        {"n_mels": 40, "lr": 1e-3, "groups": 4, "epochs": epochs},
        {"n_mels": 40, "lr": 1e-3, "groups": 8, "epochs": epochs},
    ]
    for model_params in params:
        trainer, model = get_train_instances(**model_params)
        trainer.fit(model, train_loader, test_loader)
        trainer.test(model, val_loader)


def get_dataset_loaders(batch_size: int = 32, num_workers: int = 5):
    train_dataset = BinarySpeechCommands(subset="train")
    val_dataset = BinarySpeechCommands(subset="valid")
    test_dataset = BinarySpeechCommands(subset="test")
    print("Load datasets!")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
    )
    print("Create loaders!")
    return train_loader, test_loader, val_loader


def get_train_instances(
    n_mels: int = 80, lr: float = 1e-3, groups: int = 1, epochs: int = 10
):
    logger = CSVLogger(
        "logs", name=f"model_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    model = BinarySpeechClassifier(n_mels=n_mels, lr=lr, groups=groups)

    trainer = pl.Trainer(max_epochs=epochs, logger=logger)
    return trainer, model


def collate_fn(batch):
    waves, labels = zip(*batch)
    lengths = [w.shape[0] for w in waves]
    max_length = max(lengths)
    padded_waves = [F.pad(w, (0, max_length - w.shape[0])) for w in waves]
    padded_waves = torch.stack(padded_waves)
    labels = torch.tensor(labels)
    return padded_waves, labels


if __name__ == "__main__":
    main()
