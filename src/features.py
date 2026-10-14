import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

from config import BEARING, ENVELOPE_BAND, FS

N_FFT = 8192   # zero padding for a finer frequency grid
MAX_ORDER = 12


def time_features(X):
    abs_x = np.abs(X)
    rms = np.sqrt(np.mean(X**2, axis=1))
    peak = abs_x.max(axis=1)
    return pd.DataFrame({
        "rms": rms, "peak": peak, "peak_to_peak": np.ptp(X, axis=1), "std": X.std(axis=1),
        "kurtosis": kurtosis(X, axis=1), "skewness": skew(X, axis=1),
        "crest_factor": peak / rms, "shape_factor": rms / abs_x.mean(axis=1),
        "impulse_factor": peak / abs_x.mean(axis=1),
        "clearance_factor": peak / np.mean(np.sqrt(abs_x), axis=1) ** 2,
    })


def spectral_features(X, n_bands=8):
    spectrum = np.abs(np.fft.rfft(X, axis=1)) ** 2
    freqs = np.fft.rfftfreq(X.shape[1], 1 / FS)
    total = spectrum.sum(axis=1, keepdims=True)
    edges = np.linspace(0, FS / 2, n_bands + 1)
    bands = {f"band_{int(lo / 1e3)}_{int(hi / 1e3 * 10) / 10:g}kHz":
             spectrum[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) / total[:, 0]
             for lo, hi in zip(edges[:-1], edges[1:])}
    centroid = (spectrum * freqs).sum(axis=1) / total[:, 0]
    return pd.DataFrame({"spectral_centroid": centroid, **bands})


def envelope_spectrum(X, band=ENVELOPE_BAND):
    """Demodulation: band-pass around the resonance, Hilbert envelope, then its spectrum."""
    sos = butter(4, band, btype="bandpass", fs=FS, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, X, axis=1), axis=1))
    env -= env.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(env * np.hanning(X.shape[1]), n=N_FFT, axis=1)) / X.shape[1]
    return np.fft.rfftfreq(N_FFT, 1 / FS), spec


def order_spectrum(freqs, spec, rpm, orders=np.linspace(0.5, MAX_ORDER, 460)):
    """Resample each envelope spectrum on an order axis (multiples of shaft speed)."""
    shaft = np.asarray(rpm) / 60
    return orders, np.vstack([np.interp(orders * f, freqs, s) for f, s in zip(shaft, spec)])


def envelope_features(X, rpm, harmonics=3, tolerance=0.03):
    freqs, spec = envelope_spectrum(X)
    shaft = np.asarray(rpm, dtype=float) / 60
    floor = np.median(spec[:, (freqs > 10) & (freqs < 600)], axis=1)
    feats = {}
    targets = {"BPFI": BEARING["BPFI"], "BPFO": BEARING["BPFO"], "2xBSF": 2 * BEARING["BSF"], "shaft": 1.0}
    for name, order in targets.items():
        for h in range(1, harmonics + 1):
            centre = h * order * shaft[:, None]
            window = np.abs(freqs[None] - centre) <= tolerance * centre
            feats[f"env_{name}_{h}x"] = np.log((spec * window).max(axis=1) / floor)
    return pd.DataFrame(feats)


def all_features(X, rpm):
    return pd.concat([time_features(X), spectral_features(X), envelope_features(X, rpm)], axis=1)


FEATURE_GROUPS = {
    "time": lambda cols: [c for c in cols if not c.startswith(("band_", "spectral", "env_"))],
    "spectral": lambda cols: [c for c in cols if c.startswith(("band_", "spectral"))],
    "envelope": lambda cols: [c for c in cols if c.startswith("env_")],
}
