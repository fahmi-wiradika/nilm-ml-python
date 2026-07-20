import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ─── 1. DATA PREPARATION ───────────────────────────────────────────────────

# Maps the original single class label (1-8) to 3 independent binary flags:
# [AC, TV, Lamp]. This decomposition is exact: the 8 original classes are
# precisely the 8 combinations of these 3 appliances (2^3 = 8).
CLASS_TO_BINARY = {
    1: [0, 0, 0],  # No load
    2: [1, 0, 0],  # AC only
    3: [0, 1, 0],  # TV only
    4: [0, 0, 1],  # Lamp only
    5: [1, 1, 0],  # TV + AC
    6: [0, 1, 1],  # TV + Lamp
    7: [1, 0, 1],  # AC + Lamp
    8: [1, 1, 1],  # TV + Lamp + AC
}

LOAD_NAMES = ["AC", "TV", "Lamp"]


def load_data(csv_path: str):
    """
    Expected CSV columns: rms, h1, h3, h5, h7, h9, target
    target values: 1 to 8 (matching the 8 load combinations)

    Returns:
        X: (N, 6) raw harmonic features
        y_binary: (N, 3) binary flags [AC, TV, Lamp]
        y_class: (N,) original class label 1-8, kept for reporting/comparison
    """
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    X = df[["rms", "h1", "h3", "h5", "h7", "h9"]].values.astype(np.float64)
    y_class = df["target"].values.astype(int)

    y_binary = np.array([CLASS_TO_BINARY[c] for c in y_class], dtype=np.float64)

    return X, y_binary, y_class


def normalize_input(X: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    """Replicates MATLAB mapminmax: scales inputs to [-1, 1]."""
    return (2 * X / x_max) - 1


# No target normalization needed here: binary flags are already in {0, 1},
# and the output layer uses sigmoid, which naturally outputs in [0, 1].
# This removes the mapminmax-style denormalize_output step entirely.


# ─── 2. MODEL DEFINITION ───────────────────────────────────────────────────

class LoadIdentifierNNBinary(nn.Module):
    """
    Multi-label version of the original MATLAB nntool architecture:
      Input(6) → Hidden(8, Sigmoid) → Output(3, Sigmoid)

    Each of the 3 outputs is an independent probability (AC on?, TV on?,
    Lamp on?), instead of 1 regression output decoded into a class 1-8.
    """

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(6, 8)  # 6 inputs -> 8 neurons
        self.output = nn.Linear(8, 3)  # 8 neurons -> 3 independent outputs

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # LOGSIG equivalent
        x = torch.sigmoid(self.output(x))  # independent per-load probability
        return x


# ─── 3. TRAINING ───────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, model: nn.Module, lr: float = 0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()  # Binary Cross-Entropy, per output

    def train(self, X_tensor, y_tensor, epochs: int = 1000, batch_size: int = 808):
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, epochs + 1):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

        print("\nTraining complete.")


# ─── 4. WEIGHT & BIAS EXTRACTION ───────────────────────────────────────────

def extract_weights_biases(model: nn.Module):
    """Extracts W and B from each layer, matching MATLAB nntool output format."""
    params = {}

    params["layer1_weights"] = model.hidden.weight.detach().numpy()  # shape (8, 6)
    params["layer1_bias"]    = model.hidden.bias.detach().numpy()    # shape (8,)
    params["layer2_weights"] = model.output.weight.detach().numpy()  # shape (3, 8)
    params["layer2_bias"]    = model.output.bias.detach().numpy()    # shape (3,)

    print("\n=== Layer 1 Weights (8x6) ===")
    print(np.round(params["layer1_weights"], 4))
    print("\n=== Layer 1 Bias (8,) ===")
    print(np.round(params["layer1_bias"], 4))
    print("\n=== Layer 2 Weights (3x8) ===")
    print(np.round(params["layer2_weights"], 4))
    print("\n=== Layer 2 Bias (3,) ===")
    print(np.round(params["layer2_bias"], 4))

    return params


# ─── 5. INFERENCE ──────────────────────────────────────────────────────────

def predict(model: nn.Module, X_raw: np.ndarray, x_max: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Returns an (N, 3) binary array: [AC, TV, Lamp] for each row,
    thresholded at 0.5 by default.
    """
    model.eval()
    X_norm = normalize_input(X_raw, x_max)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        y_prob = model(X_tensor).numpy()

    return (y_prob >= threshold).astype(int)


def binary_to_label(flags: np.ndarray) -> str:
    """Converts a [AC, TV, Lamp] binary row into a readable string, e.g. 'AC+Lamp'."""
    active = [name for name, flag in zip(LOAD_NAMES, flags) if flag == 1]
    return "+".join(active) if active else "No load"


# ─── 6. MAIN ───────────────────────────────────────────────────────────────

def main():
    # --- Load data ---
    BASE_DIR = Path(__file__).resolve().parent
    csv_data = BASE_DIR / "data.csv"
    X, y_binary, y_class = load_data(csv_data)

    # Max values per input feature, extracted from the CSV data
    x_max = X.max(axis=0)

    # --- Normalize inputs only (targets are already binary 0/1) ---
    X_norm = normalize_input(X, x_max)

    # --- Convert to tensors ---
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_binary, dtype=torch.float32)

    # --- Build & train model ---
    model = LoadIdentifierNNBinary()
    trainer = Trainer(model, lr=0.01)
    trainer.train(X_tensor, y_tensor, epochs=1000)

    # --- Extract weights and biases ---
    params = extract_weights_biases(model)

    # --- Test prediction on 8 random rows ---
    random_indices = np.random.choice(len(X), size=8, replace=False)
    sample = X[random_indices]
    predictions = predict(model, sample, x_max)
    actual_binary = y_binary[random_indices].astype(int)
    actual_class = y_class[random_indices]

    print("\n=== Sample Predictions ===")
    print(f"{'Actual Class':<14}{'Actual [AC,TV,Lamp]':<22}{'Predicted [AC,TV,Lamp]':<24}{'Predicted Label':<20}Match")
    for i in range(8):
        match = "✔" if np.array_equal(actual_binary[i], predictions[i]) else "✘"
        print(
            f"{actual_class[i]:<14}"
            f"{str(actual_binary[i].tolist()):<22}"
            f"{str(predictions[i].tolist()):<24}"
            f"{binary_to_label(predictions[i]):<20}"
            f"{match}"
        )


if __name__ == "__main__":
    main()