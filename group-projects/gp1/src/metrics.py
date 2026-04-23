from collections import defaultdict
import editdistance
import torch


def ctc_decode(sequence):
    result = []
    prev = None

    for t in sequence:
        if t != prev and t != 0:  # remove repeats + blanks
            result.append(t)
        prev = t

    return result

def cer(pred, target):
    return editdistance.eval(pred, target) / max(1, len(target))

def harmonic_cer(in_domain: float, out_of_domain: float) -> float:
    if not in_domain or not out_of_domain:
        return float("nan")
    return 2 * in_domain * out_of_domain / (in_domain + out_of_domain)

def decode_batch(preds, labels, label_lengths, normalizer, input_lengths=None):
    out = []
    offset = 0
    for i in range(len(label_lengths)):
        pred_i = preds[i]
        if input_lengths is not None:
            pred_i = pred_i[: int(input_lengths[i])]
        pred_tokens = ctc_decode(pred_i.cpu().numpy())
        pred_text = normalizer.tokens2num(pred_tokens)

        length = int(label_lengths[i])
        gt_tokens = labels[offset:offset + length].cpu().numpy()
        offset += length
        gt_text = normalizer.tokens2num(gt_tokens)

        out.append((pred_text, gt_text))
    return out


def evaluate_by_speaker(
    model, dataloader, normalizer, spk_ids, in_domain_speakers, device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # metrics
    spk_cer = defaultdict(list)
    in_domain_cer = []
    out_domain_cer = []

    sample_idx = 0

    with torch.no_grad():
        for x, texts, lengths, t_lengths, _ in dataloader:
            x = x.to(device)
            lengths_dev = lengths.to(device)

            logits = model(x, lengths_dev)
            preds = torch.argmax(logits, dim=-1)

            T_out = logits.size(1)
            out_lengths = (((lengths + 1) // 2 + 1) // 2).clamp(max=T_out)

            offset = 0

            for i in range(len(t_lengths)):
                pred_tokens = preds[i, : out_lengths[i]].cpu().numpy()
                pred_tokens = ctc_decode(pred_tokens)
                pred_text = normalizer.tokens2num(pred_tokens)

                length = t_lengths[i]
                gt_tokens = texts[offset : offset + length].cpu().numpy()
                offset += length
                gt_text = normalizer.tokens2num(gt_tokens)

                # metrics
                spk = spk_ids[sample_idx]
                sample_idx += 1

                sample_cer = cer(pred_text, gt_text)

                spk_cer[spk].append(sample_cer)

                if spk in in_domain_speakers:
                    in_domain_cer.append(sample_cer)
                else:
                    out_domain_cer.append(sample_cer)

    # aggregate metrics
    spk_cer_mean = {spk: sum(vals) / len(vals) for spk, vals in spk_cer.items()}
    in_domain_mean = sum(in_domain_cer) / len(in_domain_cer) if in_domain_cer else None
    out_domain_mean = (
        sum(out_domain_cer) / len(out_domain_cer) if out_domain_cer else None
    )

    return {
        "per_speaker": spk_cer_mean,
        "in_domain": in_domain_mean,
        "out_of_domain": out_domain_mean,
    }