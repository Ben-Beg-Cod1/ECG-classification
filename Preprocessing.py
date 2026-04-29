"""
preprocess.py
-------------
Verwerkt ruwe ECG-signalen naar genormaliseerde numpy arrays (.npy).

MIT-BIH  → segmenten van 360 samples (1 seconde) rondom elke beat-annotatie
CinC2017 → volledige opnames, geresampled naar 300 Hz, gepad/geknipt naar
           vaste lengte (9000 samples = 30 seconden)

Uitvoer (in data/processed/):
  mitbih/
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
  cinc2017/
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy

Gebruik:
  python data/preprocess.py --dataset mitbih
  python data/preprocess.py --dataset cinc2017
  python data/preprocess.py --dataset all
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"

# ============================================================ MIT-BIH =======

# Klassen: N=0  S=1  V=2  F=3  Q=4
MITBIH_LABEL_MAP = {
    "N": 0, ".": 0, "L": 0, "R": 0, "e": 0, "j": 0,   # Normaal
    "A": 1, "a": 1, "J": 1, "S": 1,                    # Supraventriculair
    "V": 2, "E": 2,                                     # Ventriculair
    "F": 3,                                             # Fusie
    "f": 4, "/": 4, "Q": 4, "?": 4,                    # Onbekend / overig
}
MITBIH_WINDOW = 180       # samples voor de R-piek
MITBIH_WINDOW_POST = 180  # samples na de R-piek  → totaal 360 samples = 1 s bij 360 Hz


def preprocess_mitbih():
    """Segmenteert MIT-BIH in beat-windows en slaat op als .npy."""
    try:
        import wfdb
    except ImportError:
        print(
            "[MIT-BIH] wfdb niet gevonden. Installeer met: pip install wfdb\n"
            "          Sla MIT-BIH preprocessing over."
        )
        return

    raw_dir = RAW_DIR / "mitbih"
    if not raw_dir.exists():
        print("[MIT-BIH] Ruwe data niet gevonden. Eerst downloaden met download.py.")
        return

    out_dir = PROCESSED_DIR / "mitbih"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = sorted({p.stem for p in raw_dir.glob("*.hea")})
    print(f"[MIT-BIH] {len(records)} records gevonden.")

    all_X, all_y = [], []

    for rec_id in records:
        rec_path = str(raw_dir / rec_id)
        try:
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, "atr")
        except Exception as e:
            print(f"  [skip] {rec_id}: {e}")
            continue

        signal = record.p_signal[:, 0]  # kanaal 0 (MLII)
        signal = _normalize(signal)

        for sample, symbol in zip(annotation.sample, annotation.symbol):
            if symbol not in MITBIH_LABEL_MAP:
                continue
            start = sample - MITBIH_WINDOW
            end = sample + MITBIH_WINDOW_POST
            if start < 0 or end > len(signal):
                continue
            segment = signal[start:end]
            all_X.append(segment)
            all_y.append(MITBIH_LABEL_MAP[symbol])

    X = np.array(all_X, dtype=np.float32)[:, :, np.newaxis]  # (N, 360, 1)
    y = np.array(all_y, dtype=np.int64)

    print(f"[MIT-BIH] Totaal: {len(X)} beats | klasse-verdeling: {np.bincount(y)}")

    _split_and_save(X, y, out_dir)
    print("[MIT-BIH] Opgeslagen in", out_dir, "\n")


# =========================================================== CinC 2017 ======

CINC2017_LABEL_MAP = {"N": 0, "A": 1, "O": 2, "~": 3}
CINC2017_TARGET_FS = 300    # Hz — origineel 300 Hz, geen resample nodig
CINC2017_LENGTH = 9000      # samples → 30 seconden


def preprocess_cinc2017():
    """Laadt CinC 2017 .mat bestanden, normaliseert en pad/knip naar vaste lengte."""
    try:
        from scipy.io import loadmat
    except ImportError:
        print("[CinC 2017] scipy niet gevonden. Installeer met: pip install scipy")
        return

    import csv

    raw_dir = RAW_DIR / "cinc2017" / "training2017"
    labels_path = RAW_DIR / "cinc2017" / "REFERENCE-v3.csv"

    if not raw_dir.exists() or not labels_path.exists():
        print("[CinC 2017] Ruwe data niet gevonden. Eerst downloaden met download.py.")
        return

    out_dir = PROCESSED_DIR / "cinc2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Labels inladen
    labels = {}
    with open(labels_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                labels[row[0].strip()] = row[1].strip()

    all_X, all_y = [], []

    mat_files = sorted(raw_dir.glob("*.mat"))
    print(f"[CinC 2017] {len(mat_files)} opnames gevonden.")

    for mat_path in mat_files:
        rec_id = mat_path.stem
        label_str = labels.get(rec_id)
        if label_str not in CINC2017_LABEL_MAP:
            continue

        try:
            data = loadmat(str(mat_path))
            signal = data["val"].squeeze().astype(np.float32)
        except Exception as e:
            print(f"  [skip] {rec_id}: {e}")
            continue

        signal = _normalize(signal)
        signal = _pad_or_crop(signal, CINC2017_LENGTH)

        all_X.append(signal)
        all_y.append(CINC2017_LABEL_MAP[label_str])

    X = np.array(all_X, dtype=np.float32)[:, :, np.newaxis]  # (N, 9000, 1)
    y = np.array(all_y, dtype=np.int64)

    print(f"[CinC 2017] Totaal: {len(X)} opnames | klasse-verdeling: {np.bincount(y)}")

    _split_and_save(X, y, out_dir)
    print("[CinC 2017] Opgeslagen in", out_dir, "\n")


# ============================================================== helpers ======

def _normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalisatie per signaal."""
    mu = signal.mean()
    sigma = signal.std()
    if sigma < 1e-8:
        return signal - mu
    return (signal - mu) / sigma


def _pad_or_crop(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Knip of pad (zero-padding rechts) naar exacte lengte."""
    if len(signal) >= target_len:
        return signal[:target_len]
    pad = target_len - len(signal)
    return np.pad(signal, (0, pad), mode="constant")


def _split_and_save(X: np.ndarray, y: np.ndarray, out_dir: Path):
    """Stratified 70/15/15 split en opslaan als .npy."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    for split, (Xs, ys) in zip(
        ["train", "val", "test"],
        [(X_train, y_train), (X_val, y_val), (X_test, y_test)],
    ):
        np.save(out_dir / f"X_{split}.npy", Xs)
        np.save(out_dir / f"y_{split}.npy", ys)
        print(f"  {split}: {len(Xs)} samples")


# ================================================================= main ======

def main():
    parser = argparse.ArgumentParser(description="ECG dataset preprocessor")
    parser.add_argument(
        "--dataset",
        choices=["mitbih", "cinc2017", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.dataset in ("mitbih", "all"):
        preprocess_mitbih()
    if args.dataset in ("cinc2017", "all"):
        preprocess_cinc2017()


if __name__ == "__main__":
    main()
