import os
from typing import Literal, Optional
from tqdm import tqdm
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import Dataset


class BasicSpeechCommands(SPEECHCOMMANDS):
    def __init__(
        self,
        subset: Optional[Literal["train", "valid", "test"]] = None,
        root_path: str = "./data",
    ):
        os.makedirs(root_path, exist_ok=True)
        super().__init__(root_path, download=True)

        match subset:
            case "train":
                excludes = self.load_files("validation_list.txt") + self.load_files(
                    "testing_list.txt"
                )
                walker = set(self._walker)
                self._walker = list(walker - set(excludes))
            case "valid":
                self._walker = self.load_files("validation_list.txt")
            case "test":
                self._walker = self.load_files("testing_list.txt")

    def load_files(self, filename: str):
        filepath = os.path.join(self._path, filename)
        with open(filepath) as f:
            return [os.path.join(self._path, line.strip()) for line in f]


class BinarySpeechCommands(Dataset):
    def __init__(self, subset: Literal["train", "valid", "test"]):
        self.basic_dataset = BasicSpeechCommands(subset=subset)
        self.target_classes = {"yes": 0, "no": 1}
        self.binary_indices = self.get_binary_indexes()

    def get_binary_indexes(self):
        total = len(self.basic_dataset)
        print(f"Start dataset indexing... Total: {total} samples")
        result = []
        for i in tqdm(range(len(self.basic_dataset))):
            label = self.basic_dataset[i][2]

            if label.lower() in self.target_classes:
                result.append(i)

        return result

    def __len__(self):
        return len(self.binary_indices)

    def __getitem__(self, idx):
        new_idx = self.binary_indices[idx]
        wave, _, label, _, _ = self.basic_dataset[new_idx]
        label = self.target_classes[label]
        return wave.squeeze(0), label
