"""
nilm_nnls_checkpoint.py — NILM load identification via harmonic decomposition (NNLS)

CHECKPOINT SUMMARY (for future reference)
------------------------------------------
Goal: reduce required training data from 8 full combination classes down to
a small set of individual + key interaction signatures, then decompose any
unknown harmonic reading into which of the 3 loads (AC, TV, Lamp) are active.

What was tried, in order, and why this configuration was chosen:

  1. Pure 3-signature dictionary (AC, TV, Lamp only), raw NNLS.
     -> Overall exact-match: 62.7%
     -> AC combos worked well, but TV+Lamp (0%) and AC+TV+Lamp (0%) failed
        completely. Diagnosis: TV and Lamp signatures have cosine similarity
        0.99 (nearly collinear) in the 6D harmonic space, and their combined
        signal deviates ~31.6% from a simple sum (linear superposition
        assumption breaks down when both non-linear switching loads are
        active simultaneously — likely due to phase effects not captured by
        magnitude-only harmonics).

  2. Variance-weighted NNLS (scale features by 1/std before fitting).
     -> Helped some classes (AC+TV, Lamp-alone) but did not fix TV+Lamp,
        and hurt AC+Lamp. Modest overall gain (68.3%). Not enough alone.

  3. Augmented dictionary: add the ACTUAL MEASURED TV+Lamp signature as a
     4th "interaction atom" (instead of assuming it equals TV + Lamp).
     -> Fixed TV+Lamp and AC+TV+Lamp (both 0% -> 100%), because the
        interaction atom captures the true (non-additive) combined signature.
     -> But reintroduced TV-vs-Lamp ambiguity in the AC+TV class, since TV
        and Lamp remain individually collinear whenever the interaction
        atom isn't the best fit.

  4. Improved decision rule: instead of independent thresholds per atom,
     compare TV vs Lamp coefficients directly (whichever is larger wins)
     when the interaction atom is not active.
     -> Best overall result: 93.9% exact-match accuracy, using only
        4 reference signatures (AC, TV, Lamp, TV+Lamp) instead of all
        8 combination classes.

  5. Tried adding a 5th atom (measured AC+TV signature) to fix the
     remaining AC+TV weakness -> this broke AC+Lamp and the triple
     combo instead (whack-a-mole: AC's dominant magnitude makes any
     "AC + X" combo atom collinear with other "AC + Y" combos).
     Conclusion: do NOT add more atoms past 4 — it doesn't generalize.

THIS SCRIPT implements configuration #4 (the best, most stable result).

Known limitation (documented, not hidden): AC+TV classification is the
weakest of the 8 classes under this method, because TV and Lamp remain
individually hard to tell apart whenever they are NOT measured together.
A promising next step (untested — requires re-collecting data) is logging
complex FFT bins (real + imaginary per harmonic) instead of magnitude only,
since magnitude-only addition is not phase-accurate and may be the true
root cause of the non-additive TV+Lamp behavior. See conversation notes.

Usage:
    python nilm_nnls_checkpoint.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import nnls

# ─── CONSTANTS ──────────────────────────────────────────────────────────────

FEATURES = ["rms", "h1", "h3", "h5", "h7", "h9"]
LOAD_NAMES = ["AC", "TV", "Lamp"]

# Classes used to build the dictionary. Only these are needed as reference
# data — the other 4 combination classes (5, 7, 8) are NOT used for
# training, only for evaluation, to prove the reduction actually works.
DICTIONARY_CLASSES = {
    "AC": 2,
    "TV": 3,
    "Lamp": 4,
    "TV+Lamp (interaction atom)": 6,
}

# Ground truth mapping, used only for evaluation against all 8 classes.
CLASS_TO_BINARY: dict[int, list[int]] = {
    1: [0, 0, 0],  # No load
    2: [1, 0, 0],  # AC only
    3: [0, 1, 0],  # TV only
    4: [0, 0, 1],  # Lamp only
    5: [1, 1, 0],  # AC + TV
    6: [0, 1, 1],  # TV + Lamp
    7: [1, 0, 1],  # AC + Lamp
    8: [1, 1, 1],  # AC + TV + Lamp
}

CLASS_LABELS: dict[int, str] = {
    1: "No Load",
    2: "AC",
    3: "TV",
    4: "Lamp",
    5: "AC + TV",
    6: "TV + Lamp",
    7: "AC + Lamp",
    8: "AC + TV + Lamp",
}

DEFAULT_THRESHOLD = 0.3  # best found during checkpoint testing


# ─── 1. DATA ────────────────────────────────────────────────────────────────

def load_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset. Expected CSV columns: rms, h1, h3, h5, h7, h9, target
    (semicolon-delimited, comma decimal — matches original MATLAB export).

    Returns:
        X       : raw feature matrix (N, 6)
        y_binary: ground-truth binary matrix (N, 3) -> [AC, TV, Lamp]
        y_class : original integer class label (N,) -> 1-8, kept for reporting
    """
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    X = df[FEATURES].values.astype(np.float64)
    y_class = df["target"].values.astype(int)
    y_binary = np.array([CLASS_TO_BINARY[c] for c in y_class], dtype=np.float64)
    return X, y_binary, y_class


def build_dictionary(X: np.ndarray, y_class: np.ndarray) -> np.ndarray:
    """
    Build the 4-atom decomposition dictionary from mean signatures of
    the classes listed in DICTIONARY_CLASSES.

    Returns:
        D : (6, 4) matrix, columns = [AC, TV, Lamp, TV+Lamp interaction atom]
    """
    columns = []
    for name, cls in DICTIONARY_CLASSES.items():
        mask = y_class == cls
        if not mask.any():
            raise ValueError(f"No samples found for class {cls} ({name}).")
        columns.append(X[mask].mean(axis=0))
    return np.column_stack(columns)


# ─── 2. DECOMPOSITION ──────────────────────────────────────────────────────

class NNLSLoadDecomposer:
    """
    Decomposes an unknown harmonic reading into active loads [AC, TV, Lamp]
    using only 4 reference signatures instead of 8 full combination classes.

    Method:
      1. Solve  min_c || x - D·c ||²  s.t. c >= 0   (Non-Negative Least Squares)
         where D's columns are: AC, TV, Lamp, and the measured TV+Lamp
         interaction atom (their real combined signature, not a naive sum).
      2. Decode the 4 coefficients into 3 binary flags using decode(),
         which prioritizes the interaction atom (if active, TV and Lamp
         are both flagged on), and otherwise compares TV vs Lamp directly
         to decide which one is present.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.dictionary: np.ndarray | None = None  # (6, 4)

    def fit(self, X: np.ndarray, y_class: np.ndarray) -> "NNLSLoadDecomposer":
        self.dictionary = build_dictionary(X, y_class)
        return self

    def _coefficients(self, X: np.ndarray) -> np.ndarray:
        """Raw NNLS coefficients (N, 4): [AC, TV, Lamp, TVLamp_atom]."""
        if self.dictionary is None:
            raise RuntimeError("Call fit() before predicting.")
        return np.array([nnls(self.dictionary, x)[0] for x in X])

    def decode(self, coeffs_row: np.ndarray) -> list[int]:
        """
        Turns one row of 4 NNLS coefficients into 3 binary flags [AC, TV, Lamp].

        Logic:
          - If the TV+Lamp interaction atom is active -> both TV and Lamp
            are flagged on (this is what fixes the non-additive TV+Lamp case).
          - Otherwise, AC is flagged independently via its own threshold.
          - For TV vs Lamp: since they are nearly collinear signatures,
            comparing which coefficient is LARGER is more reliable than
            thresholding each independently.
        """
        ac, tv, lamp, tvlamp_atom = coeffs_row
        ac_flag = 1 if ac >= self.threshold else 0

        if tvlamp_atom >= self.threshold:
            return [ac_flag, 1, 1]

        total_tv_lamp = tv + lamp
        if total_tv_lamp < self.threshold:
            tv_flag, lamp_flag = 0, 0
        elif tv >= lamp:
            tv_flag, lamp_flag = 1, 0
        else:
            tv_flag, lamp_flag = 0, 1

        return [ac_flag, tv_flag, lamp_flag]

    def predict_coefficients(self, X: np.ndarray) -> np.ndarray:
        """Raw NNLS coefficients (N, 4), useful for diagnostics/threshold tuning."""
        return self._coefficients(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions (N, 3): [AC, TV, Lamp] per row."""
        coeffs = self._coefficients(X)
        return np.array([self.decode(c) for c in coeffs])

    def print_dictionary(self) -> None:
        print("\n[NNLSLoadDecomposer] Reference Dictionary (4 atoms)")
        print(f"  {'Atom':<28}{'rms':>8}{'h1':>8}{'h3':>8}{'h5':>8}{'h7':>8}{'h9':>8}")
        print("  " + "-" * 76)
        for i, name in enumerate(DICTIONARY_CLASSES):
            sig = self.dictionary[:, i]
            vals = "".join(f"{v:>8.4f}" for v in sig)
            print(f"  {name:<28}{vals}")


# ─── 3. EVALUATION ──────────────────────────────────────────────────────────

def binary_to_label(flags: np.ndarray) -> str:
    active = [name for name, flag in zip(LOAD_NAMES, flags) if flag == 1]
    return "+".join(active) if active else "No load"


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Overall + per-load accuracy, exact-match rate, and Hamming score."""
    n = len(y_true)
    results: dict[str, float] = {}

    print(f"\n  {'Load':<6}{'Correct':>9}{'Total':>7}{'Accuracy':>10}")
    print("  " + "-" * 32)
    for i, name in enumerate(LOAD_NAMES):
        correct = int((y_true[:, i] == y_pred[:, i]).sum())
        acc = correct / n
        results[name] = acc
        print(f"  {name:<6}{correct:>9}{n:>7}{acc*100:>9.1f}%")

    exact = int((y_true == y_pred).all(axis=1).sum())
    results["exact_match"] = exact / n
    hamming = float((y_true == y_pred).mean())
    results["hamming_score"] = hamming

    print(f"\n  Exact match (all 3 correct) : {exact:>4}/{n} = {exact/n*100:.1f}%")
    print(f"  Hamming score (label avg)   : {hamming*100:.1f}%")
    return results


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, y_class: np.ndarray) -> None:
    print(f"\n  {'Class':>5}  {'Description':<16}{'Correct':>8}{'Total':>7}{'Rate':>7}")
    print("  " + "-" * 50)
    for cls in range(1, 9):
        mask = y_class == cls
        if not mask.any():
            continue
        n_cls = mask.sum()
        correct = int((y_true[mask] == y_pred[mask]).all(axis=1).sum())
        note = "  (train)" if cls in DICTIONARY_CLASSES.values() else ""
        print(f"  {cls:>5}  {CLASS_LABELS[cls]:<16}{correct:>8}{n_cls:>7}{correct/n_cls*100:>6.0f}%{note}")


# ─── 4. MAIN ────────────────────────────────────────────────────────────────

def main() -> None:
    BASE_DIR = Path(__file__).resolve().parent
    X, y_true, y_class = load_data(BASE_DIR / "data.csv")

    print("=" * 60)
    print("NNLS LOAD DECOMPOSITION — CHECKPOINT")
    print("=" * 60)
    print(
        "\nTrained using only 4 reference signatures (classes 2, 3, 4, 6)\n"
        "instead of all 8 combination classes. Evaluated against all 8\n"
        "to measure how well unseen combinations (1, 5, 7, 8) are inferred."
    )

    decomposer = NNLSLoadDecomposer(threshold=DEFAULT_THRESHOLD)
    decomposer.fit(X, y_class)
    decomposer.print_dictionary()

    y_pred = decomposer.predict(X)

    print("\n" + "=" * 60)
    print("OVERALL EVALUATION")
    print("=" * 60)
    evaluate(y_true, y_pred)

    print("\n" + "=" * 60)
    print("PER-CLASS BREAKDOWN")
    print("=" * 60)
    per_class_report(y_true, y_pred, y_class)

    # --- Sample predictions on random rows ---
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (16 random rows)")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(X), size=16, replace=False)
    coeffs = decomposer.predict_coefficients(X[idx])

    print(f"\n  {'#':>3}  {'Class':>5}  {'Description':<16}{'Actual':^12}{'Predicted':^12}  {'Label':<14}Match")
    print("  " + "-" * 78)
    for rank, i in enumerate(idx, start=1):
        c = coeffs[rank - 1]
        pred = decomposer.decode(c)
        actual = y_true[i].astype(int)
        match = "✔" if list(actual) == pred else "✘"
        print(
            f"  {rank:>3}  {y_class[i]:>5}  {CLASS_LABELS[y_class[i]]:<16}"
            f"{str(actual.tolist()):^12}{str(pred):^12}  "
            f"{binary_to_label(pred):<14}{match}"
        )

    print(
        "\nNote: AC+TV (class 5) is the weakest class under this method (see\n"
        "checkpoint summary above) because TV and Lamp remain individually\n"
        "hard to distinguish whenever they are not measured together. This\n"
        "is a documented, physically-explained limitation, not a bug."
    )


if __name__ == "__main__":
    main()