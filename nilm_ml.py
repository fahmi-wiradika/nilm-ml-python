import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ─── 1. DATA PREPARATION ───────────────────────────────────────────────────

def load_data(csv_path: str):
    """
    Expected CSV columns: rms, h1, h3, h5, h7, h9, target
    target values: 1 to 8 (matching your 8 load combinations)
    """
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    X = df[["rms", "h1", "h3", "h5", "h7", "h9"]].values.astype(np.float64)
    y = df["target"].values.astype(np.float64)
    return X, y


def normalize_input(X: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    """Replicates MATLAB mapminmax: scales inputs to [-1, 1]."""
    return (2 * X / x_max) - 1


def normalize_target(y: np.ndarray, target_min=10.0, target_max=80.0) -> np.ndarray:
    """
    Scales targets (1–8) to [-1, 1] to match TANSIG output range.
    MATLAB encodes targets as 10*class, so target 1→10, 8→80.
    """
    y_scaled = y * 10.0  # match MATLAB encoding: class 1→10, class 8→80
    return 2 * (y_scaled - target_min) / (target_max - target_min) - 1


def denormalize_output(y_norm: np.ndarray, target_min=10.0, target_max=80.0, scale=10.0) -> np.ndarray:
    """Reverses normalization → returns predicted class (1–8)."""
    return ((y_norm + 1) * (target_max - target_min) / 2 + target_min) / scale


# ─── 2. MODEL DEFINITION ───────────────────────────────────────────────────

class LoadIdentifierNN(nn.Module):
    """
    Replicates MATLAB nntool architecture:
      Input(6) → Hidden(8, LOGSIG) → Output(1, TANSIG)
    """

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(6, 8)  # 6 inputs → 8 neurons
        self.output = nn.Linear(8, 1)  # 8 neurons → 1 output

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # LOGSIG
        x = torch.tanh(self.output(x))     # TANSIG
        return x


# ─── 3. TRAINING ───────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, model: nn.Module, lr: float = 0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train(self, X_tensor, y_tensor, epochs: int = 1000, batch_size: int = 808):
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(1, epochs + 1):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch).squeeze()
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
    params["layer2_weights"] = model.output.weight.detach().numpy()  # shape (1, 8)
    params["layer2_bias"]    = model.output.bias.detach().numpy()    # shape (1,)

    print("\n=== Layer 1 Weights (8x6) ===")
    print(np.round(params["layer1_weights"], 4))
    print("\n=== Layer 1 Bias (8,) ===")
    print(np.round(params["layer1_bias"], 4))
    print("\n=== Layer 2 Weights (1x8) ===")
    print(np.round(params["layer2_weights"], 4))
    print("\n=== Layer 2 Bias (1,) ===")
    print(np.round(params["layer2_bias"], 4))

    return params


# ─── 5. INFERENCE ──────────────────────────────────────────────────────────

def predict(model: nn.Module, X_raw: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    model.eval()
    X_norm = normalize_input(X_raw, x_max)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        y_norm = model(X_tensor).squeeze().numpy()

    return np.round(denormalize_output(y_norm)).astype(int)


# ─── 6. MAIN ───────────────────────────────────────────────────────────────

def main():
    # --- Load data ---
    BASE_DIR = Path(__file__).resolve().parent
    csv_data = BASE_DIR / "data.csv"
    X, y = load_data(csv_data)

    # Max values per input feature, extracted from the CSV data
    x_max = X.max(axis=0)

    # --- Normalize ---
    X_norm = normalize_input(X, x_max)
    y_norm = normalize_target(y)

    # --- Convert to tensors ---
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)

    # --- Build & train model ---
    model = LoadIdentifierNN()
    trainer = Trainer(model, lr=0.01)
    trainer.train(X_tensor, y_tensor, epochs=1000)

    # --- Extract weights and biases ---
    params = extract_weights_biases(model)

    # --- Test prediction on 8 random rows ---
    random_indices = np.random.choice(len(X), size=8, replace=False)
    sample = X[random_indices]
    predictions = predict(model, sample, x_max)
    actual = y[random_indices].astype(int)

    print("\n=== Sample Predictions ===")
    for i in range(8):
        print(f"  Actual: {actual[i]}  |  Predicted: {predictions[i]}")


if __name__ == "__main__":
    main()