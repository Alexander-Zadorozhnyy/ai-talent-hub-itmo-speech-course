import os
import sys
from typing import List

sys.path.insert(0, os.getcwd())

from src.normalizers.normalizer import Normalizer
 

class UnspittableNormalizer(Normalizer):
    def __init__(self):
        self.vocab = (
            [0, 1000]
            + list(range(1, 20))
            + [i * 10 for i in range(2, 10)]
            + [i * 100 for i in range(1, 10)]
        )
        self.reverse_vocab = {v: i for i, v in enumerate(self.vocab)}
        
    def __str__(self):
        return "UnspittableNormalizer"

    def get_vocab(self):
        return self.vocab, len(self.vocab)

    def tokens2num(self, tokens: list[int]) -> str:
        number = 0
        for token in tokens:
            symbol = self.vocab[token]
            if symbol == 1000:
                number *= 1000
            elif symbol != 0:  # Skip blank tokens
                number += symbol
        return str(number)

    def num2tokens(self, number: str | int) -> list[int]:
        n = int(number)
        major, minor = divmod(n, 1000)

        tokens = []

        tokens = self.add_tokens(major, tokens)
        tokens.append(self.reverse_vocab[1000])  # thousand separator
        tokens = self.add_tokens(minor, tokens)

        return tokens

    def add_tokens(self, num: int, tokens: List[int]) -> List[int]:
        """Helper to convert 0-999 to tokens."""
        if num >= 100:
            tokens.append(self.reverse_vocab[num // 100 * 100])
            num %= 100
        if num >= 20:
            tokens.append(self.reverse_vocab[num // 10 * 10])
            num %= 10
        if num > 0:
            tokens.append(self.reverse_vocab[num])

        return tokens


if __name__ == "__main__":
    normalizer = UnspittableNormalizer()

    tokens = normalizer.num2tokens("849905")
    print(f"Tokens for 849905: {tokens}")
    number = normalizer.tokens2num(tokens)
    print(f"Number from tokens: {number}")
