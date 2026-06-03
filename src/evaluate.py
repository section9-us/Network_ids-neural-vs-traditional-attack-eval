from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


MODEL_COLORS = {
    "Logistic Regression": "#6A51A3",
    "Random Forest": "#2CA25F",
    "Shallow MLP": "#FDD049",
    "PyTorch MLP": "#2B8CBE",
    "Deep MLP": "#F03B20",
    "Autoencoder + MLP": "#BDBDBD",
}
MODEL_HATCHES = {
    "Logistic Regression": "",
    "Random Forest": "////",
    "Shallow MLP": "\\\\\\\\",
    "PyTorch MLP": "xxxx",
    "Deep MLP": "",
    "Autoencoder + MLP": "....",
}

ATTACK_FAMILY_MAP = {
    "DDOS attack-HOIC": "DoS/DDoS",
    "DDOS attack-LOIC-UDP": "DoS/DDoS",
    "DDoS attacks-LOIC-HTTP": "DoS/DDoS",
    "DoS attacks-GoldenEye": "DoS/DDoS",
    "DoS attacks-Hulk": "DoS/DDoS",
    "DoS attacks-SlowHTTPTest": "DoS/DDoS",
    "DoS attacks-Slowloris": "DoS/DDoS",
    "FTP-BruteForce": "Brute Force",
    "SSH-Bruteforce": "Brute Force",
    "Brute Force -Web": "Brute Force",
    "Brute Force -XSS": "Web Attack",
    "SQL Injection": "Web Attack",
    "Bot": "Botnet",
    "Infilteration": "Infiltration",
    "DDoS": "DoS/DDoS",
    "Brute Force": "Brute Force",
    "Botnet": "Botnet",
    "Web Attack": "Web Attack",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.85,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.65,
            "grid.linestyle": "--",
            "hatch.linewidth": 0.7,
        }
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def tidy_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def model_order(frame: pd.DataFrame) -> list[str]:
    known = [model for model in MODEL_COLORS if model in set(frame["model"])]
    rest = sorted(set(frame["model"]) - set(known))
    return known + rest


def plot_grouped_bar(
    frame: pd.DataFrame,
    *,
    category_col: str,
    value_col: str,
    ylabel: str,
    xlabel: str,
    title: str,
    output_path: Path,
    xrotation: int,
) -> None:
    apply_publication_style()
    categories = sorted(frame[category_col].astype(str).unique())
    models = model_order(frame)
    pivot = (
        frame.assign(**{category_col: frame[category_col].astype(str)})
        .pivot_table(index=category_col, columns="model", values=value_col, aggfunc="mean")
        .reindex(index=categories, columns=models)
        .fillna(0.0)
    )

    width = min(0.12, 0.78 / max(len(models), 1))
    x_positions = range(len(categories))
    fig_width = max(8.4, 0.58 * len(categories) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.9))

    for model_index, model in enumerate(models):
        offsets = [
            x + (model_index - (len(models) - 1) / 2) * width
            for x in x_positions
        ]
        bars = ax.bar(
            offsets,
            pivot[model].to_numpy(),
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, "#8C8C8C"),
            edgecolor="#222222",
            linewidth=0.55,
            hatch=MODEL_HATCHES.get(model, ""),
        )
        for bar in bars:
            bar.set_alpha(0.96)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=8)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(categories, rotation=xrotation, ha="right" if xrotation else "center")
    ax.legend(
        ncol=min(len(models), 3),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        handlelength=1.3,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    tidy_axis(ax)
    save_figure(fig, output_path)


def attack_family_for_label(label: str) -> str:
    return ATTACK_FAMILY_MAP.get(str(label).strip(), "Other Attack")


def summarize_binary_metrics(model_name: str, y_true, y_pred) -> dict[str, float | str]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": false_negative_rate,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def attack_type_detection(model_name: str, attack_labels: pd.Series, y_true, y_pred) -> pd.DataFrame:
    rows = []
    frame = pd.DataFrame(
        {
            "attack_label": attack_labels.astype(str).to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    attack_frame = frame[frame["y_true"] == 1]

    for attack_label, group in attack_frame.groupby("attack_label"):
        total = len(group)
        detected = int((group["y_pred"] == 1).sum())
        missed = total - detected
        rows.append(
            {
                "model": model_name,
                "attack_label": attack_label,
                "total": total,
                "detected": detected,
                "missed_false_negative": missed,
                "detection_rate_recall": detected / total if total else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "attack_label"])


def attack_family_detection(model_name: str, attack_labels: pd.Series, y_true, y_pred) -> pd.DataFrame:
    rows = []
    frame = pd.DataFrame(
        {
            "attack_label": attack_labels.astype(str).to_numpy(),
            "attack_family": [attack_family_for_label(label) for label in attack_labels.astype(str)],
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    attack_frame = frame[frame["y_true"] == 1]

    for attack_family, group in attack_frame.groupby("attack_family"):
        total = len(group)
        detected = int((group["y_pred"] == 1).sum())
        missed = total - detected
        rows.append(
            {
                "model": model_name,
                "attack_family": attack_family,
                "total": total,
                "detected": detected,
                "missed_false_negative": missed,
                "detection_rate_recall": detected / total if total else 0.0,
                "false_negative_rate": missed / total if total else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "attack_family"])


def save_confusion_matrix(model_name: str, y_true, y_pred, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    safe_name = model_name.lower().replace(" ", "_")
    save_figure(plt.gcf(), output_dir / f"confusion_matrix_{safe_name}.png")


def save_attack_type_plot(attack_results: pd.DataFrame, output_dir: Path) -> None:
    if attack_results.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_grouped_bar(
        attack_results,
        category_col="attack_label",
        value_col="detection_rate_recall",
        ylabel="Detection rate / recall",
        xlabel="Attack type",
        title="Attack-type-specific Detection Rate",
        output_path=output_dir / "attack_type_detection.png",
        xrotation=28,
    )


def save_attack_family_plot(attack_family_results: pd.DataFrame, output_dir: Path) -> None:
    if attack_family_results.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_grouped_bar(
        attack_family_results,
        category_col="attack_family",
        value_col="detection_rate_recall",
        ylabel="Detection rate / recall",
        xlabel="Attack family",
        title="Attack-family Detection Rate",
        output_path=output_dir / "attack_family_detection.png",
        xrotation=15,
    )


def save_false_negative_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    plot_data = metrics.sort_values("false_negative_rate", ascending=True).copy()
    min_fnr = float(plot_data["false_negative_rate"].min())
    max_fnr = float(plot_data["false_negative_rate"].max())
    lower = max(0.0, min_fnr - 0.015)
    upper = min(1.0, max_fnr + 0.015)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for index, (_, row) in enumerate(plot_data.iterrows()):
        model = row["model"]
        ax.bar(
            index,
            row["false_negative_rate"],
            width=0.58,
            color=MODEL_COLORS.get(model, "#8C8C8C"),
            edgecolor="#222222",
            linewidth=0.65,
            hatch=MODEL_HATCHES.get(model, ""),
        )
        ax.text(
            index,
            row["false_negative_rate"] + 0.002,
            f"FNR={row['false_negative_rate']:.3f}\nFN={int(row['false_negative'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(lower, upper)
    ax.set_ylabel("False negative rate")
    ax.set_xlabel("Model")
    ax.set_title("False Negative Rate by Model (Zoomed Scale)", pad=8)
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data["model"], rotation=18, ha="right")
    tidy_axis(ax)
    save_figure(fig, output_dir / "false_negative_rate.png")


def save_model_comparison_findings(attack_results: pd.DataFrame, output_dir: Path) -> None:
    if attack_results.empty:
        return

    lines = ["# Model Comparison Findings", ""]
    for attack_label, group in attack_results.groupby("attack_label"):
        ordered = group.sort_values(
            ["detection_rate_recall", "missed_false_negative"],
            ascending=[False, True],
        )
        best = ordered.iloc[0]
        worst = ordered.iloc[-1]
        lines.append(f"## {attack_label}")
        lines.append("")
        lines.append(
            f"- Best detection: {best['model']} "
            f"({best['detection_rate_recall']:.3f} recall, {int(best['total'])} test attacks)."
        )
        lines.append(
            f"- Most missed: {worst['model']} "
            f"({int(worst['missed_false_negative'])} false negatives)."
        )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_comparison_findings.md").write_text("\n".join(lines), encoding="utf-8")


def print_report(model_name: str, y_true, y_pred) -> None:
    print(f"\n=== {model_name} ===")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"], zero_division=0))
