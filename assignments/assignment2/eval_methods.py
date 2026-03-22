import os
import json
from typing import List, Optional, Tuple
import pandas as pd
import torch
import jiwer
from tqdm import tqdm
from wav2vec2decoder import Wav2Vec2Decoder
import librosa


SAMPLE_RATE = 16000


def prepare_dataset(dataset_path: str) -> Tuple[List[dict], dict]:
    print(f"Loading dataset: {dataset_path}")

    info_path = os.path.join(dataset_path, "manifest.csv")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Manifest not found: {info_path}")

    labels = {}
    dataset = []

    # Load dataset
    df = pd.read_csv(info_path)
    for idx, row in tqdm(df.iterrows()):
        # Assuming the path in manifest is relative to the working directory (assignment2)
        audio_path = row["path"]
        label = str(row["text"]).lower().strip()

        try:
            audio_input, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            audio_input = torch.tensor(audio_input).unsqueeze(0)
            if sr != SAMPLE_RATE:
                print("Incorrect sample rate, pass...")
                continue

            dataset.append(
                {
                    "id": idx,
                    "audio": audio_input,
                    "label": label,
                }
            )
            labels[idx] = label

        except Exception as e:
            print(f"Error happened while processing {audio_path}: {e}")

    return dataset, labels


def run_model_inference(
    dataset: List[dict],
    decoder: Wav2Vec2Decoder,
    methods: list = ["greedy"],
):
    results = {}

    for method in methods:
        print(f"Running evaluation for {method=}")
        tmp_result = {}
        valid_labels, valid_preds = [], []

        for row in tqdm(dataset):
            id = row["id"]
            audio_input = row["audio"]
            label = row["label"]

            try:
                prediction = decoder.decode(audio_input, method=method)
                tmp_result[id] = prediction

                valid_labels.append(label)
                valid_preds.append(prediction)
            except NotImplementedError:
                tmp_result[id] = "[ERROR] NOT_IMPLEMENTED"
            except Exception as e:
                tmp_result[id] = f"[ERROR] {e}"
                print(f"ERROR: {e}")

        wer, cer = None, None

        if valid_preds:
            wer = jiwer.wer(valid_labels, valid_preds)
            cer = jiwer.cer(valid_labels, valid_preds)

        results[method] = dict(predictions=tmp_result, metrics={"wer": wer, "cer": cer})
        print(
            f"Method: {method}. WER: {round(wer, 2) if wer is not None else None} and CER: {round(cer, 2) if cer is not None else None}"
        )

    return results


def run_evaluation(
    dataset_path: str,
    json_save_path: str,
    methods: list,
    model_name: str = "facebook/wav2vec2-base-100h",
    lm_model_path: Optional[str] = "lm/3-gram.pruned.1e-7.arpa.gz",
    beam_width_list: Optional[List[int]] = None,
    alphas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    temperatures: Optional[List[float]] = None,
):
    if beam_width_list is None:
        beam_width_list = [3]

    if alphas is None:
        alphas = [1.0]

    if betas is None:
        betas = [1.0]

    if temperatures is None:
        temperatures = [1.0]

    if "beam_lm" not in methods and "beam_lm_rescore" not in methods:
        lm_model_path = None

    dataset, labels = prepare_dataset(
        dataset_path=dataset_path,
    )
    results = {
        "config": {
            "dataset": dataset_path,
            "model_name": model_name,
            "lm_model_path": lm_model_path,
            "methods": methods,
            "beam_width_list": beam_width_list,
            "alphas": alphas,
            "betas": betas,
            "temperatures": temperatures,
        },
        "labels": labels,
        "results": [],
    }

    for beam_width in beam_width_list:
        for alpha in alphas:
            for beta in betas:
                for temperature in temperatures:
                    print(
                        f"Running evaluation for model {model_name} (optional, {lm_model_path=}) with methods: {methods}"
                    )

                    params = {
                        "beam_width": beam_width,
                        "alpha": alpha,
                        "beta": beta,
                        "temperature": temperature,
                    }

                    decoder = Wav2Vec2Decoder(
                        model_name=model_name,
                        lm_model_path=lm_model_path,
                        **params,
                    )

                    data = run_model_inference(dataset, decoder, methods)

                    results["results"].append({"params": params, "eval": data})

    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved eval data to {json_save_path}")
    return results


if __name__ == "__main__":
    run_evaluation(
        dataset_path="./data/librispeech_test_other",
        json_save_path="./results/greedy_temperature_comparison.json",
        methods=["greedy"],
        temperatures=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    )
