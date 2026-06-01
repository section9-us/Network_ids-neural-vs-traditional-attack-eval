from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"])
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(output_dir / f"confusion_matrix_{safe_name}.png", dpi=160)
    plt.close()


def save_attack_type_plot(attack_results: pd.DataFrame, output_dir: Path) -> None:
    if attack_results.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=attack_results,
        x="attack_label",
        y="detection_rate_recall",
        hue="model",
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Detection rate / recall")
    plt.xlabel("Attack type")
    plt.title("Attack-type-specific Detection Rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "attack_type_detection.png", dpi=160)
    plt.close()


def save_attack_family_plot(attack_family_results: pd.DataFrame, output_dir: Path) -> None:
    if attack_family_results.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=attack_family_results,
        x="attack_family",
        y="detection_rate_recall",
        hue="model",
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Detection rate / recall")
    plt.xlabel("Attack family")
    plt.title("Attack-family Detection Rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "attack_family_detection.png", dpi=160)
    plt.close()


def save_false_negative_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=metrics, x="model", y="false_negative_rate")
    plt.ylim(0, 1.05)
    plt.ylabel("False negative rate")
    plt.xlabel("Model")
    plt.title("False Negative Rate by Model")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "false_negative_rate.png", dpi=160)
    plt.close()


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
