from src.normalizers.normalizer import Normalizer


class DigitCharNormalizer(Normalizer):
    def __init__(self):
        self.vocab = list("0123456789")
        self.char2idx = {
            c: i + 1 for i, c in enumerate(self.vocab)
        }  # 0 reserved for blank
        self.reverse_vocab = {i: c for c, i in self.char2idx.items()}
        
    def __str__(self):
        return "DigitCharNormalizer"

    def get_vocab(self) -> tuple[list, int]:
        return self.vocab, len(self.vocab)

    def num2tokens(self, number):
        return [self.char2idx[c] for c in str(number)]

    def tokens2num(self, tokens):
        digits = []
        prev = None

        for t in tokens:
            if t != prev and t != 0:
                digits.append(self.reverse_vocab[t])
            prev = t

        return "".join(digits) if digits else "0"
