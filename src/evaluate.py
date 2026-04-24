from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


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


def print_report(model_name: str, y_true, y_pred) -> None:
    print(f"\n=== {model_name} ===")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"], zero_division=0))
