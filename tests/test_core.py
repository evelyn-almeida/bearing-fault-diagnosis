import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import BEARING, FS, WINDOW
from src.anomaly import HealthIndex
from src.data import RECORDS
from src.features import envelope_features, envelope_spectrum, time_features

RPM = 1797


def impulsive_signal(fault_hz, n=WINDOW, seed=0):
    """Train of decaying 3 kHz bursts repeating at the fault frequency, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = 0.05 * rng.normal(size=n)
    for t0 in np.arange(0, t[-1], 1 / fault_hz):
        m = t >= t0
        x[m] += np.exp(-(t[m] - t0) * 800) * np.sin(2 * np.pi * 3000 * (t[m] - t0))
    return x


def test_record_table():
    assert len(RECORDS) == 40
    assert RECORDS["label"].value_counts().to_dict() == {"Ball": 12, "Inner race": 12, "Outer race": 12, "Normal": 4}


def test_time_features():
    x = np.vstack([np.sin(np.linspace(0, 20 * np.pi, WINDOW))])
    f = time_features(x)
    assert np.isclose(f["crest_factor"][0], np.sqrt(2), atol=0.01)
    assert f["kurtosis"][0] < 0  # a sine wave is not impulsive


def test_envelope_recovers_fault_frequency():
    bpfo = BEARING["BPFO"] * RPM / 60
    freqs, spec = envelope_spectrum(impulsive_signal(bpfo)[None])
    band = (freqs > 20) & (freqs < 500)
    assert abs(freqs[band][np.argmax(spec[0, band])] - bpfo) < 6


def test_envelope_features_point_to_the_right_fault():
    bpfi = BEARING["BPFI"] * RPM / 60
    f = envelope_features(impulsive_signal(bpfi)[None], [RPM])
    assert f["env_BPFI_1x"][0] > f["env_BPFO_1x"][0]


def test_health_index_flags_faults():
    rng = np.random.default_rng(0)
    cols = ["rms", "peak", "peak_to_peak", "std", "kurtosis"]
    normal = pd.DataFrame(np.abs(rng.normal(1, 0.1, (200, 5))), columns=cols)
    faulty = normal.copy()
    faulty["kurtosis"] += 2
    hi = HealthIndex().fit(normal, np.repeat([0, 1, 2, 3], 50))
    assert np.mean(hi.score(faulty) > hi.threshold) > 0.95
    assert np.mean(hi.score(normal) > hi.threshold) < 0.05
