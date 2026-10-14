import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from config import BEARING, CLASSES, FIG_DIR, FS

BLUE, ORANGE, GREEN, RED, GRAY = "#1f4e79", "#e07a1f", "#2a9d8f", "#c0392b", "#8a8a8a"
CLASS_COLORS = dict(zip(CLASSES, [GREEN, BLUE, ORANGE, RED]))
MODEL_COLORS = {"Time-domain statistics": GRAY, "Envelope fault frequencies (physics)": GREEN,
                "All hand-crafted features": BLUE, "Envelope order spectrum + logistic regression": ORANGE}
FAULT_LINES = {"BPFI": ("Inner race", "BPFI"), "BPFO": ("Outer race", "BPFO"), "2xBSF": ("Ball", "2×BSF")}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def waveforms(examples, kurt):
    fig, axes = plt.subplots(4, 1, figsize=(13, 8.5), sharex=True)
    t = np.arange(len(next(iter(examples.values())))) / FS * 1e3
    for ax, (label, x) in zip(axes, examples.items()):
        ax.plot(t, x, color=CLASS_COLORS[label], lw=0.7)
        ax.set_ylabel("g")
        ax.text(0.01, 0.92, f"{label}   ·   kurtosis = {kurt[label]:.1f}", transform=ax.transAxes,
                fontweight="bold", va="top", bbox=dict(boxstyle="round", fc="white", ec=GRAY))
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Raw vibration (drive-end accelerometer): defects create periodic impacts",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "01_waveforms")


def envelope_spectra(spectra, freqs, rpm):
    fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True)
    for ax, (label, spec) in zip(axes, spectra.items()):
        ax.plot(freqs, spec, color=CLASS_COLORS[label], lw=1.2)
        for key, (fault, text) in FAULT_LINES.items():
            order = 2 * BEARING["BSF"] if key == "2xBSF" else BEARING[key]
            for h in range(1, 4):
                f = h * order * rpm[label] / 60
                ax.axvline(f, color=CLASS_COLORS[fault], ls="--", lw=1, alpha=0.7)
                if label == "Normal":
                    ax.text(f, 1.02, f"{h}×{text}" if h > 1 else text, color=CLASS_COLORS[fault], fontsize=7,
                            rotation=90, ha="center", va="bottom", transform=ax.get_xaxis_transform())
        ax.set(ylabel="Amplitude", xlim=(0, 520))
        ax.text(0.99, 0.9, label, transform=ax.transAxes, ha="right", va="top", fontweight="bold",
                color=CLASS_COLORS[label])
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Envelope spectra: each defect rings at the frequency predicted by bearing kinematics",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "02_envelope_spectra")


def feature_distributions(features, labels):
    cols = [("kurtosis", "Kurtosis", False), ("rms", "RMS (g)", True),
            ("env_BPFI_1x", "Envelope peak at BPFI (log ratio)", False),
            ("env_BPFO_1x", "Envelope peak at BPFO (log ratio)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (col, title, log) in zip(axes, cols):
        data = [features.loc[labels == c, col] for c in CLASSES]
        box = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False, tick_labels=CLASSES)
        for patch, c in zip(box["boxes"], CLASSES):
            patch.set(facecolor=CLASS_COLORS[c], alpha=0.6)
        ax.set(title=title, yscale="log" if log else "linear")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Condition indicators separate the classes, but no single one is enough",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_feature_distributions")


def feature_space(xy, meta):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for c in CLASSES:
        m = (meta["label"] == c).to_numpy()
        axes[0].scatter(xy[m, 0], xy[m, 1], s=6, color=CLASS_COLORS[c], alpha=0.7, label=c)
    axes[0].set_title("Coloured by fault type")
    axes[0].legend(markerscale=3)
    codes = meta["label"] + " " + meta["severity"]
    for i, code in enumerate(sorted(codes.unique())):
        m = (codes == code).to_numpy()
        axes[1].scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.7, color=plt.cm.tab10(i % 10), label=code)
    axes[1].set_title("Coloured by physical bearing (fault type + size)")
    axes[1].legend(markerscale=3, fontsize=8, ncol=2)
    for ax in axes:
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
    fig.suptitle("Feature space (t-SNE): segments cluster by the individual bearing they came from",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_feature_space")


def protocols(results):
    table = results.pivot(index="protocol", columns="model", values="balanced_accuracy")
    order = ["Random segment split", "Leave one load out", "Unseen fault size"]
    table = table.loc[order, list(MODEL_COLORS)]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    width = 0.8 / table.shape[1]
    for i, (model, color) in enumerate(MODEL_COLORS.items()):
        bars = ax.bar(np.arange(len(order)) + (i - (table.shape[1] - 1) / 2) * width, table[model], width,
                      color=color, label=model)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(0.25, color="black", ls=":", lw=1)
    ax.text(2.45, 0.26, "chance", fontsize=8)
    ax.set_xticks(range(len(order)), ["Random segment split\n(what most tutorials report)",
                                      "Leave one motor load out\n(same bearings, new load)",
                                      "Unseen fault size\n(new physical bearings)"])
    ax.set(ylim=(0, 1.1), ylabel="Balanced accuracy (4 classes)",
           title="The same models, three ways to test them: accuracy depends on what is held out")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save(fig, "05_evaluation_protocols")


def confusion(matrices):
    fig, axes = plt.subplots(1, len(matrices), figsize=(6.2 * len(matrices), 5))
    for ax, (title, m) in zip(np.atleast_1d(axes), matrices.items()):
        im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=25, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{m.values[i, j]:.0%}", ha="center", va="center",
                        color="white" if m.values[i, j] > 0.6 else "black")
        ax.set(xlabel="Predicted", ylabel="True", title=title)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8)
    save(fig, "06_confusion_matrices")


def noise_robustness(noise):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for model, g in noise.groupby("model", sort=False):
        color = MODEL_COLORS[model]
        ax.plot(g["snr_label"], g["balanced_accuracy"], marker="o", lw=2.5, color=color, label=model)
    ax.axhline(0.25, color="black", ls=":", lw=1)
    ax.set(xlabel="Signal-to-noise ratio of the test signal", ylabel="Balanced accuracy",
           ylim=(0, 1.05), title="Robustness to sensor noise (trained on clean data, tested on an unseen load)")
    ax.legend(loc="lower left")
    save(fig, "07_noise_robustness")


def anomaly_detection(scores, labels, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(scores.min(), np.percentile(scores, 99.5), 60)
    for c in CLASSES:
        axes[0].hist(scores[labels == c], bins=bins, color=CLASS_COLORS[c], alpha=0.55, density=True, label=c)
    axes[0].axvline(threshold, color="black", ls="--", lw=1.5, label="Alarm threshold")
    axes[0].set(xlabel="Health index relative to the alarm threshold", ylabel="Density",
                title="Health index on unseen loads (trained on normal data only)")
    axes[0].legend()
    is_normal = labels == "Normal"
    for c in CLASSES[1:]:
        m = is_normal | (labels == c)
        fpr, tpr, _ = roc_curve(labels[m] != "Normal", scores[m])
        auc = np.trapezoid(tpr, fpr)
        axes[1].plot(fpr, tpr, color=CLASS_COLORS[c], lw=2.5, label=f"{c} (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color=GRAY, ls=":")
    axes[1].set(xlabel="False alarm rate", ylabel="Detection rate", title="ROC: fault vs. healthy")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "08_anomaly_detection")


def alarm_rate_by_load(table):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(table.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(table.shape[1]), [f"{c} hp" for c in table.columns])
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontweight="bold",
                    color="white" if v > 0.6 else "black")
    ax.set(xlabel="Motor load of the unseen test condition",
           title="Alarm rate: faults always flagged, false alarms depend on the load")
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    save(fig, "09_alarm_rate_by_load")


def feature_importance(importance, groups, top=15):
    imp = importance.sort_values().tail(top)
    colors = {"time": BLUE, "spectral": GREEN, "envelope": ORANGE}
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp.index, imp.values, color=[colors[groups[c]] for c in imp.index])
    for g, c in colors.items():
        ax.barh([], [], color=c, label=f"{g} features")
    ax.set(xlabel="Mean decrease in impurity", title="What the random forest relies on")
    ax.legend(loc="lower right")
    save(fig, "10_feature_importance")
