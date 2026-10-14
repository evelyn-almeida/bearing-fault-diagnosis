import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate

from config import DATA_DIR, WINDOW

NOMINAL_RPM = [1797, 1772, 1750, 1730]   # motor load 0, 1, 2, 3 hp
DE = "12k_Drive_End_Bearing_Fault_Data"


def _spec():
    rows = [("Normal", "-", load, 97 + load, f"Normal/{97 + load}_Normal_{load}.mat") for load in range(4)]
    for label, folder, sizes in [("Ball", "B", [("007", 118), ("014", 185), ("021", 222)]),
                                 ("Inner race", "IR", [("007", 105), ("014", 169), ("021", 209)])]:
        rows += [(label, f"0.{s}", load, n + load, f"{DE}/{folder}/{s}/{n + load}_{load}.mat")
                 for s, n in sizes for load in range(4)]
    for s, n, sub, tag in [("007", 130, "@6/", "@6"), ("014", 197, "", "@6"), ("021", 234, "@6/", "")]:
        rows += [("Outer race", f"0.{s}", load, n + load, f"{DE}/OR/{s}/{sub}{n + load}{tag}_{load}.mat")
                 for load in range(4)]
    return pd.DataFrame(rows, columns=["label", "severity", "load", "number", "path"])


RECORDS = _spec()


def load_signal(row):
    mat = loadmat(DATA_DIR / row.path)
    x = mat[f"X{row.number:03d}_DE_time"].ravel().astype(float)
    if row.label == "Normal":
        # The normal baseline was recorded at 48 kHz: anti-alias and bring it to 12 kHz like the faults
        x = decimate(x, 4, ftype="fir", zero_phase=True)
    rpm_key = f"X{row.number:03d}RPM"
    rpm = float(mat[rpm_key].ravel()[0]) if rpm_key in mat else NOMINAL_RPM[row.load]
    return x, rpm


def build_dataset(window=WINDOW):
    """Non-overlapping segments of every record, with metadata."""
    signals, meta = [], []
    for row in RECORDS.itertuples():
        x, rpm = load_signal(row)
        n = len(x) // window
        signals.append(x[: n * window].reshape(n, window))
        meta += [{"label": row.label, "severity": row.severity, "load": row.load, "rpm": rpm,
                  "record": row.number, "segment": i} for i in range(n)]
    return np.vstack(signals), pd.DataFrame(meta)
