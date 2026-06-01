from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from data import align_features_for_inference, to_binary_label
from models import load_mlp_checkpoint, predict_mlp_proba


MODEL_FILES = {
    "logistic_regression": "logistic_regression.pkl",
    "random_forest": "random_forest.pkl",
    "shallow_mlp": "shallow_mlp.pt",
    "pytorch_mlp": "pytorch_mlp.pt",
    "deep_mlp": "deep_mlp.pt",
    "autoencoder_mlp": "autoencoder_mlp.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict benign vs suspicious flow-level traffic.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to a flow CSV for prediction.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FILES),
        default="pytorch_mlp",
        help="Saved model to use for prediction.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--label-column", default=None, help="Optional label column for evaluation context.")
    parser.add_argument("--output", type=Path, default=Path("reports/demo_predictions.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_names = json.loads((args.artifact_dir / "feature_names.json").read_text(encoding="utf-8"))
    with (args.artifact_dir / "scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)

    raw = pd.read_csv(args.csv)
    features, labels = align_features_for_inference(raw, feature_names, label_column=args.label_column)
    x = scaler.transform(features).astype("float32")

    model_path = args.artifact_dir / MODEL_FILES[args.model]
    if args.model.endswith("_mlp"):
        model, threshold = load_mlp_checkpoint(model_path)
        probabilities = predict_mlp_proba(model, x)
    else:
        with model_path.open("rb") as file:
            model = pickle.load(file)
        threshold = 0.5
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(x)[:, 1]
        else:
            probabilities = model.predict(x)

    predictions = (probabilities >= threshold).astype(int)
    output = pd.DataFrame(
        {
            "model": args.model,
            "attack_probability": probabilities,
            "prediction": ["suspicious" if pred == 1 else "benign" for pred in predictions],
            "binary_prediction": predictions,
        }
    )

    if labels is not None:
        output.insert(0, "original_attack_label", labels.to_numpy())
        output["binary_true"] = to_binary_label(labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
