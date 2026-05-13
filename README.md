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
├── nilm_ml.py          # Original monolithic migration (reference)
├── requirements.txt    # Dependencies
├── data.csv            # Training data (gitignored — add your own)
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

Place your `data.csv` in the project root (it is gitignored to protect research data), then run:

```bash
python nilm_ml.py
```

This will:
1. Load and normalize the dataset
2. Train the 6→8→1 network for 1000 epochs
3. Print Layer 1 and Layer 2 weights and biases
4. Run predictions on 8 random samples and print actual vs. predicted class

### Expected Output

```
Epoch  100 | Loss: 0.024312
Epoch  200 | Loss: 0.008741
...
Epoch 1000 | Loss: 0.000183

Training complete.

=== Layer 1 Weights (8x6) ===
[[ 14.2109  31.9712 -67.1798 ...]
 ...

=== Sample Predictions ===
  Actual: 3  |  Predicted: 3
  Actual: 7  |  Predicted: 7
  ...
```

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

---

## License

MIT License — see [LICENSE](LICENSE) for details.

© 2026 [Fahmi Wahyu Wiradika](https://github.com/fahmi-wiradika) and contributors. All rights reserved.
