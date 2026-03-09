## Assignment 1. Digital Signal Processing - [20 pts]

### LogMelFilterBanks

- [**Implemented**](melbanks.py) a PyTorch layer (inherit a class from `torch.nn.Module`) for extraction of **logarithms of Mel-scale Filterbank energies** using basic `torch` operations (`torch.stft`, `torch.matmul`, `torch.log`) and so on.

- Wrote tests to check some possible values and compare with standart non log `torchaudio.transforms.MelSpectrogram` realization. To run them use:

    ```bash
    cd ./assignments/assignment1;
    python3 -m pytest .\tests\test_melbanks.py
    ```
    ![**Results**](imgs/test_melbank_results.png)

### CNN Training
