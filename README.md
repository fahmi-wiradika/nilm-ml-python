# NILM-ML-Project

A Python reimplementation of the Neural Network load identification algorithm originally developed in MATLAB and deployed on an STM32F407VGTX microcontroller as part of a bachelor research project at the **Electronic Engineering Polytechnic Institute of Surabaya (EEPIS)**.

> **Research title:** Non-Intrusive Household Load Identification Using FFT Harmonic Analysis and Backpropagation Neural Network on STM32F407  
> **Author:** Fahmi Wahyu Wiradika  
> **Original implementation:** MATLAB nntool + C firmware on STM32F407VGTX  
> **This repository:** Python/PyTorch migration of the NN training and inference pipeline

---

## Background

Electricity waste in households is a common and largely invisible problem. Consumers typically only discover they have used too much power when their token expires or the monthly bill arrives. Appliances left on standby, chargers left plugged in, and lights left on during the day all contribute silently to excessive consumption.

This project addresses that problem by building a **Smart Energy Meter** that can identify which household appliances are currently connected and consuming power — not by installing a separate sensor on each appliance, but by analyzing the **harmonic signature** of the combined current waveform at a single measurement point. This approach falls under the field of **Non-Intrusive Load Monitoring (NILM)**.

The meter measures voltage and current using AMC1300 isolation amplifiers connected to an STM32F407VGTX microcontroller. The microcontroller runs FFT on the sampled signals at 3200 samples/second to extract harmonic components, then feeds those harmonics into a pre-trained Neural Network to classify which combination of loads is active. Results are shown on an LCD touchscreen and sent via Bluetooth.

This repository contains the Python migration of the NN training pipeline: the part originally implemented in MATLAB for training and in C for inference on the microcontroller.

---

## Current Project Stage

The project has moved beyond the initial MATLAB-to-Python migration. At the current stage, the repository contains three complementary modeling approaches that are being compared and developed:

- `nilm_ml.py`: the baseline regression-style neural network. It follows the original MATLAB-inspired idea of predicting one class label from 1 to 8 using a 6→8→1 architecture.
- `nilm_ml_binary.py`: the current improved model formulation. Instead of predicting a single class number, it predicts three independent binary outputs for AC, TV, and Lamp. This makes the task more natural because each appliance state is modeled directly and the final combination can be reconstructed from those flags.
- `nilm_decomp_ml.py`: an experimental decomposition-based approach using NNLS and a small dictionary of reference signatures. This script is still in the development stage and is mainly a research topic, because the physical behavior of switching components such as the TV and Lamp is not strictly linear. Their combined harmonic signature cannot always be modeled as a simple sum of the individual signatures.

### Why `nilm_ml_binary.py` is an improvement over `nilm_ml.py`

Compared with `nilm_ml.py`, the binary model is a more suitable formulation for this problem. The original script treats load identification as a single multiclass regression problem, while the binary version decomposes the problem into three appliance-level decisions. This provides a more interpretable output structure, better matches the physical idea of identifying which loads are active, and avoids forcing all possible combinations into one discrete class index. In short, it is a more flexible and physically meaningful approach for multi-load identification.

---

## How the System Works

```
220V AC Line
     │
     ▼
┌──────────────────────────────────────┐
│         AMC1300 Sensor Board         │
│  Voltage divider + Current shunt     │
│  Signal conditioning (OPA376)        │
│  Zero Crossing Detector (LM2903D)    │
└──────────────┬───────────────────────┘
               │  ADC input (0–3.3V)
               ▼
┌──────────────────────────────────────┐
│       STM32F407VGTX (168 MHz)        │
│                                      │
│  1. ADC sampling @ 3200 Hz           │
│  2. FFT (CMSIS radix-2)              │
│     → RMS, H1, H3, H5, H7, H9       │
│  3. Neural Network inference (C)     │
│     → Load class 1–8                 │
│  4. Power calculation                │
│     → P, Q, S, THD                   │
│  5. Display on LCD ILI9341           │
│  6. Transmit via Bluetooth HC-05     │
└──────────────────────────────────────┘
```

### Signal Processing Pipeline

The STM32 samples voltage and current at **3200 Hz**, satisfying Shannon's theorem for harmonics up to the 9th order (9 × 50 Hz = 450 Hz). Raw ADC samples are transformed from the time domain to the frequency domain using the **FFT radix-2 algorithm** via the ARM CMSIS DSP library. This produces the harmonic spectrum from which the following six features are extracted per measurement cycle:

| Feature | Description |
|---------|-------------|
| `rms` | RMS value of load current |
| `h1` | Fundamental harmonic (50 Hz) |
| `h3` | 3rd harmonic (150 Hz) |
| `h5` | 5th harmonic (250 Hz) |
| `h7` | 7th harmonic (350 Hz) |
| `h9` | 9th harmonic (450 Hz) |

These six values form the input vector to the Neural Network.

---

## Neural Network Architecture

The NN was originally designed and trained in **MATLAB nntool** and replicates a classic backpropagation architecture:

```
Input Layer       Hidden Layer        Output Layer
  (6 neurons)      (8 neurons)         (1 neuron)

   rms  ──┐
   h1   ──┤                          ┌── tanh ──► denorm ──► Load Class (1–8)
   h3   ──┼──► [W1 · x + b1] ──► sigmoid ──► [W2 · x + b2] ──┘
   h5   ──┤
   h7   ──┤
   h9   ──┘
```

| Layer | Size | Activation | MATLAB equivalent |
|-------|------|------------|-------------------|
| Input | 6 | — | — |
| Hidden | 8 | Sigmoid | LOGSIG |
| Output | 1 | Tanh | TANSIG |

### Normalization

Before entering the network, inputs are normalized to `[-1, 1]` using `mapminmax` from MATLAB, replicated as:

```
x_norm = (2 * x / x_max) - 1
```

The `x_max` values are fixed constants derived from the maximum measured values in the training dataset:

```python
x_max = [0.5781, 0.5096, 0.2095, 0.0889, 0.0775, 0.0639]
#         rms     h1      h3      h5      h7      h9
```

Output from the tanh layer is denormalized back to load class 1–8:

```
load_class = round( ((y_norm + 1) × (80 - 10) / 2 + 10) / 10 )
```

This encoding maps target 1 → 10, target 8 → 80 to match MATLAB's training convention.

---

## Dataset

Training data consists of **808 samples** (101 per load combination) across **8 classes** representing all possible combinations of three household loads:

| Class | Load Combination |
|-------|-----------------|
| 1 | No load |
| 2 | Air Conditioner only |
| 3 | LED TV only |
| 4 | Energy-Efficient Lamp only |
| 5 | LED TV + Air Conditioner |
| 6 | LED TV + Energy-Efficient Lamp |
| 7 | Air Conditioner + Energy-Efficient Lamp |
| 8 | LED TV + Energy-Efficient Lamp + Air Conditioner |

Each appliance has a distinct harmonic fingerprint. For example, the Air Conditioner (inductive motor load) shows energy concentrated at the fundamental frequency with near-zero higher harmonics, while the LED TV and Energy-Efficient Lamp (non-linear switching loads) produce significant 3rd, 5th, 7th, and 9th harmonics. The NN exploits these differences to classify combinations even when their total RMS currents are similar.

### CSV Format

```
rms,h1,h3,h5,h7,h9,target
0,0,0,0,0,0,1
0.1763,0.1601,0,0,0,0,2
0.2071,0.0983,0.0992,0.0707,0.0473,0.017,3
...
0.5499,0.4939,0.1699,0.0756,0.0187,0.0523,8
```

> The CSV uses semicolon (`;`) as delimiter and comma (`,`) as decimal separator to match the original MATLAB export format. This is handled automatically in `loader.py`.

---

## Pre-trained Weights

The original weights obtained after 534 training iterations in MATLAB (achieving R ≈ 1 on training, validation, and test sets) are documented in the research for direct embedding into the STM32 firmware:

**Layer 1 Weights (8×6):**
```
Neuron 1:  [ 14.2109,  31.9712, -67.1798, -65.9171, -28.8300,   4.0190]
Neuron 2:  [-37.6200, -57.1581,  42.6421, -12.6324, -14.7227,   8.9370]
Neuron 3:  [ -1.3751,   4.7549,  44.2802, -13.7080, -19.3646,  -2.7440]
Neuron 4:  [-20.8764,  -2.9745,  -1.0278,  -1.5516,  -9.7267,  -0.4373]
Neuron 5:  [244.7532, 201.8466, -30.4454,  11.7708,   3.9845,  -1.9483]
Neuron 6:  [ -4.3426, 138.1010, -24.2581, -52.2111, -57.3235, -19.2858]
Neuron 7:  [ 11.3929, -33.2200,  25.6788,   5.1466,   3.6771,   6.4989]
Neuron 8:  [-62.7838, -38.6550, -78.1346,  19.2386,  12.0053, -15.6732]
```

**Layer 1 Bias (8,):**
```
[-50.3135, 6.3582, -19.6014, 9.2935, -126.4218, -83.9288, -22.2309, -41.3666]
```

**Layer 2 Weights (1×8):**
```
[-8.9663, -0.28768, 0.60199, 1.025e-07, 0.75204, 8.5285, -1.919e-08, -0.3143]
```

**Layer 2 Bias:**
```
[0.14384]
```

These are embedded as C arrays in the STM32 firmware for real-time inference.

---

## Project Structure

```
NILM-ML-Project/
├── nilm_ml.py             # Baseline regression-style classifier (class 1–8)
├── nilm_ml_binary.py      # Improved multi-label binary classifier for AC/TV/Lamp
├── nilm_decomp_ml.py      # Experimental NNLS decomposition approach
├── requirements.txt       # Dependencies
├── data.csv               # Training data (gitignored — add your own)
└── README.md
```

> A modular refactor of this codebase (config-driven, multi-model registry, exportable weights) is planned as a follow-up. See the architecture proposal in the project notes.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/fahmi-wiradika/NILM-ML-Project.git
cd NILM-ML-Project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Place your `data.csv` in the project root (it is gitignored to protect research data), then run one of the available scripts:

```bash
python nilm_ml.py
python nilm_ml_binary.py
python nilm_decomp_ml.py
```

These scripts cover the current development stage of the project:
1. `nilm_ml.py` trains the baseline 6→8→1 network and prints class-based predictions.
2. `nilm_ml_binary.py` trains the improved multi-label model and predicts AC/TV/Lamp activity as binary flags.
3. `nilm_decomp_ml.py` explores decomposition-based identification and is still under research due to the non-linear behavior of switching loads.

### Expected Output

#### 1) Baseline model: `nilm_ml.py`

```text
(.venv) PS C:\Fahmi\Onboarding\nilm-ml-python> python .\nilm_ml.py
Epoch  100 | Loss: 0.023218
Epoch  200 | Loss: 0.010416
Epoch  300 | Loss: 0.008320
Epoch  400 | Loss: 0.006986
Epoch  500 | Loss: 0.006124
Epoch  600 | Loss: 0.005545
Epoch  700 | Loss: 0.005113
Epoch  800 | Loss: 0.004749
Epoch  900 | Loss: 0.004410
Epoch 1000 | Loss: 0.004083

Training complete.

=== Layer 1 Weights (8x6) ===
[[-2.3081 -2.167   0.6977  2.0946 -0.0558 -0.262 ]
 ...]

=== Sample Predictions ===
  Actual: 4  |  Predicted: 4
  Actual: 7  |  Predicted: 7
  Actual: 1  |  Predicted: 1
  Actual: 6  |  Predicted: 6
  Actual: 6  |  Predicted: 6
  Actual: 6  |  Predicted: 6
  Actual: 8  |  Predicted: 8
  Actual: 3  |  Predicted: 3
```

This script outputs a single predicted class in the range 1–8.

#### 2) Improved binary model: `nilm_ml_binary.py`

```text
(.venv) PS C:\Fahmi\Onboarding\nilm-ml-python> python .\nilm_ml_binary.py
Epoch  100 | Loss: 0.392771
Epoch  200 | Loss: 0.202685
Epoch  300 | Loss: 0.132555
Epoch  400 | Loss: 0.099895
Epoch  500 | Loss: 0.080258
Epoch  600 | Loss: 0.067072
Epoch  700 | Loss: 0.057438
Epoch  800 | Loss: 0.049943
Epoch  900 | Loss: 0.043689
Epoch 1000 | Loss: 0.038281

Training complete.

=== Sample Predictions ===
Actual Class  Actual [AC,TV,Lamp]   Predicted [AC,TV,Lamp]  Predicted Label     Match
4             [0, 0, 1]             [0, 0, 1]               Lamp                ✔
6             [0, 1, 1]             [0, 1, 1]               TV+Lamp             ✔
8             [1, 1, 1]             [1, 1, 1]               AC+TV+Lamp          ✔
6             [0, 1, 1]             [0, 1, 1]               TV+Lamp             ✔
5             [1, 1, 0]             [1, 1, 0]               AC+TV               ✔
7             [1, 0, 1]             [1, 0, 1]               AC+Lamp             ✔
7             [1, 0, 1]             [1, 0, 1]               AC+Lamp             ✔
5             [1, 1, 0]             [1, 1, 0]               AC+TV               ✔
```

This script outputs three independent binary flags for AC, TV, and Lamp.

#### 3) Experimental decomposition model: `nilm_decomp_ml.py`

```text
(.venv) PS C:\Fahmi\Onboarding\nilm-ml-python> python .\nilm_decomp_ml.py
============================================================
NNLS LOAD DECOMPOSITION — CHECKPOINT
============================================================

Trained using only 4 reference signatures (classes 2, 3, 4, 6)
instead of all 8 combination classes. Evaluated against all 8
to measure how well unseen combinations (1, 5, 7, 8) are inferred.

[NNLSLoadDecomposer] Reference Dictionary (4 atoms)
  Atom                             rms      h1      h3      h5      h7      h9
  ----------------------------------------------------------------------------
  AC                            0.1733  0.1675  0.0000  0.0000  0.0000  0.0000
  TV                            0.2070  0.1079  0.0928  0.0719  0.0475  0.0216
  Lamp                          0.2674  0.1710  0.1194  0.0643  0.0497  0.0436
  TV+Lamp (interaction atom)    0.3791  0.2820  0.1854  0.0603  0.0048  0.0193

============================================================
OVERALL EVALUATION
============================================================

  Load    Correct  Total  Accuracy
  --------------------------------
  AC          808    808    100.0%
  TV          761    808     94.2%
  Lamp        759    808     93.9%

  Exact match (all 3 correct) :  759/808 = 93.9%
  Hamming score (label avg)   : 96.0%

============================================================
PER-CLASS BREAKDOWN
============================================================

  Class  Description      Correct  Total   Rate
  --------------------------------------------------
      1  No Load              101    101   100%
      2  AC                   101    101   100%  (train)
      3  TV                   100    101    99%  (train)
      4  Lamp                  95    101    94%  (train)
      5  AC + TV               60    101    59%
      6  TV + Lamp            101    101   100%  (train)
      7  AC + Lamp            100    101    99%
      8  AC + TV + Lamp       101    101   100%

============================================================
SAMPLE PREDICTIONS (16 random rows)
============================================================

    #  Class  Description        Actual    Predicted    Label         Match
  ------------------------------------------------------------------------------
    1      4  Lamp             [0, 0, 1]   [0, 0, 1]    Lamp          ✔
    2      1  No Load          [0, 0, 0]   [0, 0, 0]    No load       ✔
    3      1  No Load          [0, 0, 0]   [0, 0, 0]    No load       ✔
    4      6  TV + Lamp        [0, 1, 1]   [0, 1, 1]    TV+Lamp       ✔
    5      6  TV + Lamp        [0, 1, 1]   [0, 1, 1]    TV+Lamp       ✔
    6      6  TV + Lamp        [0, 1, 1]   [0, 1, 1]    TV+Lamp       ✔
    7      5  AC + TV          [1, 1, 0]   [1, 1, 0]    AC+TV         ✔
    8      7  AC + Lamp        [1, 0, 1]   [1, 0, 1]    AC+Lamp       ✔
    9      7  AC + Lamp        [1, 0, 1]   [1, 0, 1]    AC+Lamp       ✔
   10      6  TV + Lamp        [0, 1, 1]   [0, 1, 1]    TV+Lamp       ✔
   11      4  Lamp             [0, 0, 1]   [0, 0, 1]    Lamp          ✔
   12      1  No Load          [0, 0, 0]   [0, 0, 0]    No load       ✔
   13      7  AC + Lamp        [1, 0, 1]   [1, 0, 1]    AC+Lamp       ✔
   14      8  AC + TV + Lamp   [1, 1, 1]   [1, 1, 1]    AC+TV+Lamp    ✔
   15      7  AC + Lamp        [1, 0, 1]   [1, 0, 1]    AC+Lamp       ✔
   16      2  AC               [1, 0, 0]   [1, 0, 0]    AC            ✔

Note: AC+TV (class 5) is the weakest class under this method (see
checkpoint summary above) because TV and Lamp remain individually
hard to distinguish whenever they are not measured together. This
is a documented, physically-explained limitation, not a bug.

```

This script shows the NNLS-based decomposition logic and is still considered a research-stage approach, especially for cases where TV and Lamp interact non-linearly.

---

## From MATLAB to Python: Migration Notes

| Concern | MATLAB (original) | Python (this repo) |
|---------|-------------------|-------------------|
| Training algorithm | `trainlm` (Levenberg-Marquardt) | Adam optimizer |
| Hidden activation | `logsig` | `torch.sigmoid` |
| Output activation | `tansig` | `torch.tanh` |
| Input normalization | `mapminmax` | Manual: `(2*x/x_max) - 1` |
| Target encoding | Class × 10, range [10, 80] | Same |
| Weight extraction | nntool GUI export | `model.hidden.weight.detach().numpy()` |
| Inference target | STM32 C firmware | Python (portable to C via weight export) |

The most significant difference is the optimizer: MATLAB's default `trainlm` uses Levenberg-Marquardt, a second-order method that converges very quickly for small networks. Adam is a first-order method but is more hardware-agnostic and sufficient for this network size. Convergence may require more epochs than the original 534 iterations.

---

## Hardware Context

The full research system includes:

- **Microcontroller:** STM32F407VGTX (ARM Cortex-M4, 168 MHz, hardware FPU, CMSIS-DSP)
- **Sensors:** AMC1300 isolation amplifiers (Texas Instruments), ±250 mVpp differential input
- **Current sensing:** 15 mΩ / 6W shunt resistor, 15A fuse protection
- **Voltage sensing:** 1.68 MΩ / 1 KΩ resistor divider, clamped to 250 mVpp
- **Signal conditioning:** OPA376 op-amp summing circuit + Schottky diode clamp
- **Zero Crossing Detection:** LM2903D dual comparator, REF2033 1.65V reference
- **Display:** LCD TFT 2.6" ILI9341 resistive touch (SPI)
- **Communication:** Bluetooth HC-05 (USART3)
- **Storage:** SD card (SDIO)
- **Power:** MPM-10-5 AC/DC rectifier → 5V, isolated via ROE-0505S for sensor high-side

The Python NN training produces weights and biases that are hardcoded as C arrays into the STM32 firmware for real-time inference — no floating-point ML framework runs on the microcontroller.

---

## References

1. Morello, R. (2017). *A Smart Power Meter to Monitor Energy Flow in Smart Grids.* University Mediterranea of Reggio Calabria.
2. Hutoro, K. (2012). *Design of Smart Meter to Monitor and Identify Household Energy Consumption Using Backpropagation Neural Network.* Institute of Technology of Tokyo University.
3. Nagano, H. (2017). *Application of the Time-Frequency Analysis using Wavelet Transform to Harmonic Analysis in the Power Conversion System.* Kobe City College of Technology.

---

## Full Research Documentation

The complete research is documented chapter by chapter at:

- [Fahmi's Bachelor Research](https://fahmi-wiradika.github.io/projects/academic/research-project/bachelor-research)
- [GitHub Repository](https://github.com/fahmi-wiradika/nilm-ml-python)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

© 2026 [Fahmi Wahyu Wiradika](https://github.com/fahmi-wiradika) and contributors. All rights reserved.
