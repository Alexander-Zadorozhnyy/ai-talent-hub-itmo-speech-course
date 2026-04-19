import os
import sys
from typing import List
from num2words import num2words
import re

sys.path.insert(0, os.getcwd())

from src.normalizers.normalizer import Normalizer


class RussianAlphabetWordNormalizer(Normalizer):
    def __init__(self):
        self.vocab = list(" абвгдежзийклмнопрстуфхцчшщъыьэюя")

        # index mappings (0 reserved for CTC blank)
        self.char2idx = {c: i + 1 for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

        # parsing dictionaries
        self.units = {
            "ноль": 0,
            "один": 1,
            "одна": 1,
            "два": 2,
            "две": 2,
            "три": 3,
            "четыре": 4,
            "пять": 5,
            "шесть": 6,
            "семь": 7,
            "восемь": 8,
            "девять": 9,
            "десять": 10,
            "одиннадцать": 11,
            "двенадцать": 12,
            "тринадцать": 13,
            "четырнадцать": 14,
            "пятнадцать": 15,
            "шестнадцать": 16,
            "семнадцать": 17,
            "восемнадцать": 18,
            "девятнадцать": 19,
        }

        self.tens = {
            "двадцать": 20,
            "тридцать": 30,
            "сорок": 40,
            "пятьдесят": 50,
            "шестьдесят": 60,
            "семьдесят": 70,
            "восемьдесят": 80,
            "девяносто": 90,
        }

        self.hundreds = {
            "сто": 100,
            "двести": 200,
            "триста": 300,
            "четыреста": 400,
            "пятьсот": 500,
            "шестьсот": 600,
            "семьсот": 700,
            "восемьсот": 800,
            "девятьсот": 900,
        }

        self.thousands = {"тысяча", "тысячи", "тысяч"}

    def __str__(self):
        return "RussianAlphabetWordNormalizer"

    def get_vocab(self):
        return self.vocab, len(self.vocab) + 1  # + blank

    def _clean(self, text: str) -> str:
        text = text.lower()
        text = text.replace("ё", "е")  # normalize
        text = text.replace("-", " ")
        text = re.sub(r"[^а-я ]", "", text)
        return text.strip()

    def num2tokens(self, number: str | int) -> List[int]:
        text = num2words(int(number), lang="ru")
        text = self._clean(text)
        return [self.char2idx[c] for c in text]

    def tokens2num(self, tokens: List[int]) -> str:
        chars = []
        prev = None

        # CTC decode
        for t in tokens:
            if t != 0 and t != prev:
                chars.append(self.idx2char.get(t, ""))
            prev = t

        text = "".join(chars).strip()

        # split words
        words = text.split()

        total = 0
        current = 0

        for w in words:
            if w in self.hundreds:
                current += self.hundreds[w]
            elif w in self.tens:
                current += self.tens[w]
            elif w in self.units:
                current += self.units[w]
            elif w in self.thousands:
                total += current * 1000
                current = 0

        total += current

        return str(total if total > 0 else 0)


if __name__ == "__main__":
    normalizer = RussianAlphabetWordNormalizer()

    tokens = normalizer.num2tokens("849905")
    print(f"Tokens for 849905: {tokens}")
    number = normalizer.tokens2num(tokens)
    print(f"Number from tokens: {number}")
