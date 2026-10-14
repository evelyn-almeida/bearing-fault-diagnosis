"""Bearing fault detection and diagnosis on CWRU data: python run.py"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import CLASSES, RESULTS_DIR, SEED
from src import plots
from src.anomaly import HealthIndex
from src.data import build_dataset
from src.features import FEATURE_GROUPS, all_features, envelope_spectrum
from src.models import FeatureModel, model_zoo

SEVERITIES = ["0.007", "0.014", "0.021"]


def splits(meta):
    """Train/test index pairs for each evaluation protocol."""
    idx = np.arange(len(meta))
    tr, te = train_test_split(idx, test_size=0.2, stratify=meta["label"], random_state=SEED)
    yield "Random segment split", tr, te
    for load in range(4):
        yield "Leave one load out", idx[meta["load"] != load], idx[meta["load"] == load]
    normal = meta["label"] == "Normal"
    for i, sev in enumerate(SEVERITIES):
        test = (meta["severity"] == sev) | (normal & (meta["load"] == i + 1))
        yield "Unseen fault size", idx[~test], idx[test]


def add_noise(X, snr_db, rng):
    power = np.mean(X**2, axis=1, keepdims=True) / 10 ** (snr_db / 10)
    return X + rng.normal(size=X.shape) * np.sqrt(power)


def main():
    clock = time.time()
    X, meta = build_dataset()
    y, rpm = meta["label"].to_numpy(), meta["rpm"].to_numpy()
    feats = all_features(X, rpm)
    print(f"{len(X)} segments, {feats.shape[1]} features [{time.time() - clock:.0f}s]")

    rows, preds = [], {}
    for protocol, tr, te in splits(meta):
        for model in model_zoo():
            pred = model.fit(X[tr], rpm[tr], y[tr]).predict(X[te], rpm[te])
            rows.append({"protocol": protocol, "model": model.name,
                         "balanced_accuracy": balanced_accuracy_score(y[te], pred)})
            if model.name == "All hand-crafted features":
                preds.setdefault(protocol, []).append((y[te], pred))
    folds = pd.DataFrame(rows)
    results = folds.groupby(["protocol", "model"], sort=False)["balanced_accuracy"].mean().reset_index()
    print(f"protocols done [{time.time() - clock:.0f}s]")

    rng = np.random.default_rng(SEED)
    tr, te = np.where(meta["load"] != 3)[0], np.where(meta["load"] == 3)[0]
    snrs = [None, 20, 10, 5, 0, -5]
    noise_rows = []
    for model in model_zoo():
        model.fit(X[tr], rpm[tr], y[tr])
        for snr in snrs:
            Xt = X[te] if snr is None else add_noise(X[te], snr, rng)
            noise_rows.append({"model": model.name, "snr_label": "clean" if snr is None else f"{snr} dB",
                               "balanced_accuracy": balanced_accuracy_score(y[te], model.predict(Xt, rpm[te]))})
    noise = pd.DataFrame(noise_rows)
    print(f"noise done [{time.time() - clock:.0f}s]")

    scores = np.empty(len(meta))
    is_alarm = np.zeros(len(meta), dtype=bool)
    for load in range(4):
        healthy = (y == "Normal") & (meta["load"] != load).to_numpy()
        detector = HealthIndex().fit(feats[healthy], meta.loc[healthy, "load"])
        test = (meta["load"] == load).to_numpy()
        scores[test] = detector.score(feats[test]) - detector.threshold
        is_alarm[test] = scores[test] > 0
    faults = y != "Normal"
    detection = pd.DataFrame({"label": y, "load": meta["load"], "alarm": is_alarm})
    by_load = detection.pivot_table(index="label", columns="load", values="alarm").loc[CLASSES]
    anomaly = {"auc": roc_auc_score(faults, scores), "false_alarm_rate": is_alarm[~faults].mean(),
               "detection_rate": is_alarm[faults].mean()}

    RESULTS_DIR.mkdir(exist_ok=True)
    folds.to_csv(RESULTS_DIR / "protocol_folds.csv", index=False, float_format="%.4f")
    table = results.pivot(index="model", columns="protocol", values="balanced_accuracy")
    (RESULTS_DIR / "protocols.md").write_text(table.round(3).to_markdown())
    noise.to_csv(RESULTS_DIR / "noise_robustness.csv", index=False, float_format="%.4f")
    by_load.to_csv(RESULTS_DIR / "alarm_rate_by_load.csv", float_format="%.4f")
    (RESULTS_DIR / "anomaly.json").write_text(json.dumps(anomaly, indent=2))

    plots.setup_style()
    examples, kurt, spectra, rpms = {}, {}, {}, {}
    for c in CLASSES:
        m = (meta["label"] == c) & (meta["load"] == 0) & meta["severity"].isin(["-", "0.007"])
        examples[c] = X[m.to_numpy()][3][:1200]
        kurt[c] = feats.loc[m, "kurtosis"].median()
        freqs, spec = envelope_spectrum(X[m.to_numpy()])
        spectra[c], rpms[c] = spec.mean(axis=0), meta.loc[m, "rpm"].iloc[0]
    plots.waveforms(examples, kurt)
    plots.envelope_spectra(spectra, freqs, rpms)
    plots.feature_distributions(feats, y)
    xy = TSNE(perplexity=40, random_state=SEED, init="pca").fit_transform(StandardScaler().fit_transform(feats))
    plots.feature_space(xy, meta)
    plots.protocols(results)

    def matrix(protocol):
        true = np.concatenate([t for t, _ in preds[protocol]])
        pred = np.concatenate([p for _, p in preds[protocol]])
        return pd.DataFrame(confusion_matrix(true, pred, labels=CLASSES, normalize="true"), CLASSES, CLASSES)
    plots.confusion({"Leave one load out": matrix("Leave one load out"),
                     "Unseen fault size": matrix("Unseen fault size")})
    plots.noise_robustness(noise)
    plots.anomaly_detection(scores, y, 0.0)
    plots.alarm_rate_by_load(by_load)

    forest = FeatureModel("All", ["time", "spectral", "envelope"]).fit(X, rpm, y)
    groups = {c: g for g, f in FEATURE_GROUPS.items() for c in f(feats.columns)}
    plots.feature_importance(pd.Series(forest.model.feature_importances_, feats.columns), groups)

    print(f"\ndone [{time.time() - clock:.0f}s]\n")
    print(table.round(3).to_string())
    print(noise.pivot(index="model", columns="snr_label", values="balanced_accuracy")[
        ["clean"] + [f"{s} dB" for s in snrs[1:]]].round(3).to_string())
    print(anomaly)
    print(by_load.round(3).to_string())


if __name__ == "__main__":
    main()
