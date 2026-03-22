import heapq
import math
from typing import List, Literal, Tuple, Union

import kenlm
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa

# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
        self,
        model_name="facebook/wav2vec2-base-100h",
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width=3,
        alpha=1.0,
        beta=1.0,
        temperature=1.0,
    ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = "".join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, " ").strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        tokens = torch.argmax(log_probs, dim=-1).tolist()

        result = []
        prev = None

        for curr in tokens:
            if curr != self.blank_token_id and curr != prev:
                result.append(curr)
            prev = curr

        return self._ids_to_text(result)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        time_step_num, vocab_size = log_probs.shape

        # beam: {prefix: (prob_blank, prob_non_blank)}
        prev_beam = {(): (0.0, float("-inf"))}

        for time_step in range(time_step_num):
            probs_t = log_probs[time_step].tolist()
            curr_beam = self.create_beam(prev_beam, probs_t, vocab_size)

            # Truncate curr beams
            best_hyps = heapq.nlargest(
                self.beam_width,
                curr_beam.items(),
                key=lambda x: _log_add(x[1][0], x[1][1]),
            )
            prev_beam = dict(best_hyps)

        result = [
            (list(prefix), _log_add(p_b, p_nb))
            for prefix, (p_b, p_nb) in prev_beam.items()
        ]
        result.sort(key=lambda x: x[1], reverse=True)

        if return_beams:
            return result

        return self._ids_to_text(result[0][0])

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")

        log_probs = torch.log_softmax(logits, dim=-1)
        time_step_num, vocab_size = log_probs.shape

        # beam: {prefix: (prob_blank, prob_non_blank)}
        prev_beam = {(): (0.0, float("-inf"))}

        for time_step in range(time_step_num):
            probs_t = log_probs[time_step].tolist()
            curr_beam = self.create_beam(prev_beam, probs_t, vocab_size)

            # Truncate with LM scoring
            scored_beam = self.score_beam(
                curr_beam, log_p_acoustic=True, result_type="full"
            )

            best_scored = heapq.nlargest(
                self.beam_width, scored_beam, key=lambda x: x[2]
            )
            prev_beam = {prefix: scores for prefix, scores, _ in best_scored}

        # Output rescored result
        result = self.score_beam(prev_beam, log_p_acoustic=True, result_type="prefix")
        result.sort(key=lambda x: x[1], reverse=True)
        return self._ids_to_text(result[0][0])

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        result = self.score_beam(beams, log_p_acoustic=False, result_type="text")
        return sorted(result, key=lambda x: x[1], reverse=True)[0][0]

    def create_beam(self, old_beam: dict, probs_t, v_size: int):
        new_beam = {}

        for prefix, (p_b, p_nb) in old_beam.items():
            p_total = _log_add(p_b, p_nb)

            # Expand with blank
            if prefix not in new_beam:
                new_beam[prefix] = (float("-inf"), float("-inf"))

            new_p_b, new_p_nb = new_beam[prefix]
            new_beam[prefix] = (
                _log_add(new_p_b, p_total + probs_t[self.blank_token_id]),
                new_p_nb,
            )

            # Expand with non-blank
            for char in range(v_size):
                if char == self.blank_token_id:
                    continue

                prob_char = probs_t[char]
                expanded_prefix = prefix + (char,)

                if len(prefix) > 0 and char == prefix[-1]:
                    if expanded_prefix not in new_beam:
                        new_beam[expanded_prefix] = (float("-inf"), float("-inf"))

                    expanded_p_b, expanded_p_nb = new_beam[expanded_prefix]
                    new_beam[expanded_prefix] = (
                        expanded_p_b,
                        _log_add(expanded_p_nb, p_b + prob_char),
                    )

                    # Collapse -- same character with no blank token
                    new_p_b, new_p_nb = new_beam[prefix]
                    new_beam[prefix] = (new_p_b, _log_add(new_p_nb, p_nb + prob_char))
                else:
                    # Add new character
                    if expanded_prefix not in new_beam:
                        new_beam[expanded_prefix] = (float("-inf"), float("-inf"))

                    expanded_p_b, expanded_p_nb = new_beam[expanded_prefix]
                    new_beam[expanded_prefix] = (
                        expanded_p_b,
                        _log_add(expanded_p_nb, p_total + prob_char),
                    )

        return new_beam

    def score_beam(
        self,
        beam: Union[dict, list],
        result_type: Union[Literal["full"], Literal["prefix"], Literal["text"]],
        log_p_acoustic: bool = False,
    ):
        scored_beam = []

        beam_data = beam.items() if isinstance(beam, dict) else beam
        for prefix, acoustic in beam_data:
            if log_p_acoustic:
                acoustic = _log_add(acoustic[0], acoustic[1])

            text = self._ids_to_text(prefix)
            lm_score = self.lm_model.score(text) if text else 0
            num_words = len(text.split()) if text else 0

            total_score = acoustic + self.alpha * lm_score + self.beta * num_words
            match result_type:
                case "prefix":
                    scored_beam.append((prefix, total_score))
                case "text":
                    scored_beam.append((text, total_score))
                case "full":
                    scored_beam.append((prefix, beam[prefix], total_score))

        return scored_beam

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------


def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = librosa.load(audio_path, sr=16000)
    audio_input = torch.tensor(audio_input).unsqueeze(0)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in [
        "beam_lm_rescore"
    ]:  # ,"greedy", "beam" , "beam_lm", "beam_lm_rescore"
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    # Reference transcripts are lowercase to match the evaluation manifests.
    # examples/ clips are for quick debugging only — use data/librispeech_test_other/
    # and data/earnings22_test/ for all reported metrics.
    test_samples = [
        (
            "examples/sample1.wav",
            "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance",
        ),
        (
            "examples/sample2.wav",
            "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months",
        ),
        (
            "examples/sample3.wav",
            "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin",
        ),
        (
            "examples/sample4.wav",
            "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom",
        ),
        (
            "examples/sample5.wav",
            "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes",
        ),
        (
            "examples/sample6.wav",
            "at this time all participants are in a listen only mode",
        ),
        (
            "examples/sample7.wav",
            "the increase was mainly attributable to the net increase in the average size of our fleets",
        ),
        (
            "examples/sample8.wav",
            "operating surplus is a non cap financial measure which is defined as fully in our press release",
        ),
    ]

    decoder = Wav2Vec2Decoder(
        lm_model_path="./lm/3-gram.pruned.1e-7.arpa", temperature=1
    )  # set lm_model_path for Tasks 4+

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)
