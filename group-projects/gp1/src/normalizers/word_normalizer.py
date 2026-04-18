import os
import sys
from typing import List
from num2words import num2words
import re

sys.path.insert(0, os.getcwd())

from src.normalizers.normalizer import Normalizer


class RussianWordNormalizer(Normalizer):
    def __init__(self):
        # dictionaries for parsing
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

        vocab_words = set()

        # all possible numbers
        for n in range(1, 1000):
            words = self._clean(num2words(n, lang="ru")).split()
            vocab_words.update(words)

        # generate thousands (this produces "две", "одна")
        for n in range(1, 1000):
            words = self._clean(num2words(n * 1000, lang="ru")).split()
            vocab_words.update(words)

        vocab_words.update(self.thousands)

        vocab_words = sorted(vocab_words)

        # reserve 0 for CTC blank
        self.word2idx = {w: i + 1 for i, w in enumerate(vocab_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        
    def __str__(self):
        return "RussianWordNormalizer"

    def get_vocab(self):
        return list(self.word2idx.keys()), len(self.word2idx) + 1  # + blank

    def num2tokens(self, number: str | int) -> List[int]:
        text = num2words(int(number), lang="ru")
        text = self._clean(text)

        words = text.split()
        return [self.word2idx[w] for w in words]

    def tokens2num(self, tokens: List[int]) -> str:
        words = []
        prev = None

        # CTC collapse
        for t in tokens:
            if t != 0 and t != prev:
                if t in self.idx2word:
                    words.append(self.idx2word[t])
            prev = t

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

    def _clean(self, text: str) -> str:
        text = text.lower()
        text = text.replace("-", " ")
        text = re.sub(r"[^а-яё ]", "", text)
        return text.strip()
    
if __name__ == "__main__":
    normalizer = RussianWordNormalizer()

    tokens = normalizer.num2tokens("849905")
    print(f"Tokens for 849905: {tokens}")
    number = normalizer.tokens2num(tokens)
    print(f"Number from tokens: {number}")