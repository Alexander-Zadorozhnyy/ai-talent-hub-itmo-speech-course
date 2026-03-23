# Assignment 2. ASR Decoding - [20 pts]

## Part 1 — CTC Decoding

### Task 1. `greedy_decode` [[line 77]](wav2vec2decoder.py#77)

Was implemented `greedy_decode`, which simply finds best CTC path. Evaluation result on `data/librispeech_test_other/` dataset:

| Implementation | WER | CER |
|---|---|---|
| Reference | 10.4% | 3.5% |
| MyImplementation | 11.2% | 3.8% |

### Task 2. `beam_search_decode` [[line 100]](wav2vec2decoder.py#100)

Was implemented `beam_search_decode`, which perform beam search decoding without using any LM. Evaluation result on `data/librispeech_test_other/` dataset:

| Implementation | beam_width | WER | CER | Inference Time (1 decode, sec)
|---|---|---|---|---|
| Reference | - | 9.9% | 3.4% | - |
| MyImplementation | 1 | 11.2% | 3.8% | 0.36 |
| MyImplementation | 3 | 11.1% | 3.7% | 0.4 |
| MyImplementation | **10** | **11%** | **3.7%** | **0.57** |
| MyImplementation | 50 | 11% | 3.7% | 1.5 |

Heatmap:
![**Heatmap**](imgs/beam_width_analysis.png)

So, as we can see WER getting lower with higher beam_width, but at the same time inference time also growing.

### Task 3. Temperature scaling for acoustic model outputs

Results of changing `T ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0}` on `data/librispeech_test_other/` using **greedy decoding**:

| Temperature | WER | CER |
|---|---|---|
| 0.5 | 11.2% | 3.8% |
| 0.8 | 11.2% | 3.8% |
| 1.0 | 11.2% | 3.8% |
| 1.2 | 11.2% | 3.8% |
| 1.5 | 11.2% | 3.8% |
| 2.0 | 11.2% | 3.8% |

As we can see from results temperature do not impact on WER and CER by **greedy decoding** method (argmax is invariant to scaling).

## Part 2 — Language Model Integration

### Task 4. `beam_search_with_lm` [[line 146]](wav2vec2decoder.py#146)

Was implemented `beam_search_with_lm`, which perform beam search decoding + **3-gram LM**. Evaluation result on `data/librispeech_test_other/` dataset:

| Implementation | alpha | beta | WER | CER |
|---|---|---|---|---|
| Reference | - | - | 9.7% | 3.4% |
| MyBestImplementationv1 | 1.0 | 1.0 | 10.9% | 3.76% |
| MyBestImplementationv2 | 0.05 | 0.5 | 11.02% | 3.76% |

Heatmap:
![**Heatmap**](imgs/alpha_beta_analysis.png)

### Task 5. `beam_search_with_lm` using **4-gram LM**

Comparison 3-gram vs 4-gram LM using v1 and v2 best params from Task4.

LM | alpha | beta | WER | CER |
|---|---|---|---|---|
| 3-gram | 1.0 | 1.0 | **10.9%** | **3.76%** |
| 4-gram | 1.0 | 1.0 | 11.04% | 3.76% |
| 3-gram | 0.05 | 0.5 | 11.02% | 3.76% |
| 4-gram | 0.05 | 0.5 | **11%** | **3.76%** |

### Task 6. `lm_rescore` [[line186]](wav2vec2decoder.py#186)

Was implemented `beam_search_with_lm_rescore`, which perform beam search decoding + **3-gram LM** and later rescore. Evaluation result on `data/librispeech_test_other/` dataset:

| Implementation | alpha | beta | WER | CER |
|---|---|---|---|---|
| Reference | - | - | 9.6% | 3.3% |
| MyBestImplementationv1 | 0.01 | 1.0 | 11.09% | 3.76% |

Heatmap:
![**Heatmap**](imgs/alpha_beta_lm_rescore_analysis.png)

Conclusion:
Rescoring is highly more stable at high alphas: WER = 11.6% vs 20.8% (alpha=5.0) for shallow fusion, because sf not just rearrange, but also use it during the search.

### Task 7a. Between domain comparison

| Method | LibriSpeech WER | LibriSpeech CER | Earnings22 WER | Earnings22 CER |
|---|---|---|---|---|
| Greedy | 11.2% | 3.812% | 54.9% | 25.57% |
| Beam search | 11.07% | 3.769% | 55.1% | 25.4% |
| Beam + 3-gram (shallow fusion) | 10.99 | 3.764 | 55.18% | 25.44% |
| Beam + 3-gram (rescoring) | 11.09% | 3.769% | 55.18% | 25.47% |

Big gap between domains: ~11% vs ~55% WER (LibriSpeech vs Earnings22). This decrease in quality occurs because the acoustic model is trained on LibriSpeech audiobooks and does not generalize well to financial speech. Languale Model, which were trained at LibriSpeech too, slightly decrease the result at Earnings22 (produces more "literary" corrections rather than financial).

### Task 7b  

Conclusions:

- Greedy always stays the same unchanged (argmax is invariant to scaling). Beam+LM deteriorates with increasing T: % → %;
- A high T smooths out the acoustic distribution => LM has more influence on the final score => but LM is trained on another dataset (LibriSpeech) => more mistakes in financial vocabulary
- Comparison with LibriSpeech (Task 3): the LM methods worked correctly at T=1, since both the acoustic model and the LM were trained on the same domain. On the other hand, the acoustic model has never seen Earnings22 and the error rate increases with increasing temperature.
