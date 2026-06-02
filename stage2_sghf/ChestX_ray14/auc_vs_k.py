import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def _ensure_prob_np(y_score_np: np.ndarray) -> np.ndarray:
    if y_score_np.min() < 0.0 or y_score_np.max() > 1.0:
        y_score_np = 1.0 / (1.0 + np.exp(-y_score_np))
    return y_score_np

def _safe_auc_binary(y_true_1d: np.ndarray, y_prob_1d: np.ndarray):
    if y_true_1d.sum() == 0 or (1 - y_true_1d).sum() == 0:
        return np.nan
    return roc_auc_score(y_true_1d, y_prob_1d)

def mean_auc_vs_label_cardinality(
    y_true: np.ndarray,
    y_score: np.ndarray,
    bins=(1, 2, 3, 99)
):
    """
      ks_labels: ['k=1','k=2','k=3','k≥4']
      mean_auc_per_bin: [float...]
      counts_per_bin: [int...]
    """
    y_score = _ensure_prob_np(y_score.copy())
    k = y_true.sum(axis=1).astype(int)

    max_fixed = bins[-2] if len(bins) >= 2 else bins[-1]
    ks_labels = [rf"$k={b}$" for b in bins[:-1]] + [rf"$k\geq {max_fixed+1}$"]
    masks = [(k == b) for b in bins[:-1]] + [(k >= (max_fixed + 1))]

    mean_auc_per_bin, counts_per_bin = [], []
    for mask in masks:
        idx = np.where(mask)[0]
        counts_per_bin.append(int(idx.size))
        if idx.size < 2:
            mean_auc_per_bin.append(np.nan)
            continue
        auc_c = []
        for c in range(y_true.shape[1]):
            auc_c.append(_safe_auc_binary(y_true[idx, c], y_score[idx, c]))
        mean_auc = float(np.nanmean(auc_c)) if np.any(~np.isnan(auc_c)) else np.nan
        mean_auc_per_bin.append(mean_auc)

    return ks_labels, mean_auc_per_bin, counts_per_bin

def plot_mean_auc_vs_k(
    ks_labels, mean_auc_per_bin, counts_per_bin,
    out_png="./figs/nih14_mean_auc_vs_k.png",
    title=r"Mean AUC vs Label Cardinality ($k$)"
):
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(ks_labels))
    ys = np.array(mean_auc_per_bin, dtype=float)

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(xs, ys, marker='o', linewidth=2)
    for i, (x, y) in enumerate(zip(xs, ys)):
        txt = "NaN" if np.isnan(y) else f"{y:.3f}"
        plt.text(x, (0.5 if np.isnan(y) else y), f"{txt}\n(n={counts_per_bin[i]})",
                 ha='center', va='bottom', fontsize=9)
    plt.xticks(xs, ks_labels)
    plt.ylim(0.0, 1.0)
    plt.xlabel(r"Label Cardinality $k$")
    plt.title(r"Mean AUC vs Label Cardinality ($k$)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_mean_auc_vs_k_csv(
    ks_labels, mean_auc_per_bin, counts_per_bin,
    out_csv="./figs/nih14_mean_auc_vs_k.csv"
):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k_bin", "mean_auc", "num_samples"])
        for kbin, aucv, n in zip(ks_labels, mean_auc_per_bin, counts_per_bin):
            w.writerow([kbin, "" if np.isnan(aucv) else f"{aucv:.6f}", n])
