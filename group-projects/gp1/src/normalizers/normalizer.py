from abc import ABC, abstractmethod


class Normalizer(ABC):
    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def get_vocab(self) -> tuple[list, int]:
        pass

    @abstractmethod
    def tokens2num(self, tokens: list) -> str:
        pass

    @abstractmethod
    def num2tokens(self, number: str | int) -> list[int]:
        pass
