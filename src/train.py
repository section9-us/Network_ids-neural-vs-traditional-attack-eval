from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data import dataset_profile, generate_sample_flows, load_flow_csv, prepare_data
from evaluate import (
    attack_family_detection,
    attack_type_detection,
    print_report,
    save_attack_family_plot,
    save_attack_type_plot,
    save_confusion_matrix,
    save_false_negative_plot,
    save_model_comparison_findings,
    summarize_binary_metrics,
)
from models import predict_mlp, save_mlp_checkpoint, train_neural_model


DEFAULT_NEURAL_MODELS = "shallow_mlp,pytorch_mlp,deep_mlp,autoencoder_mlp"


def safe_model_name(model_name: str) -> str:
    return (
        model_name.lower()
        .replace(" + ", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare traditional ML and PyTorch MLP for flow-level IDS."
    )
    parser.add_argument("--csv", type=Path, help="Path to a flow-level IDS CSV file.")
    parser.add_argument("--label-column", default=None, help="Name of the label column.")
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate and use a toy flow dataset under data/sample_flows.csv.",
    )
    parser.add_argument("--max-rows", type=int, default=50000, help="Optional stratified sample size.")
    parser.add_argument("--epochs", type=int, default=20, help="PyTorch MLP training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="PyTorch MLP batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="PyTorch MLP learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="PyTorch MLP hidden dimension.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Autoencoder latent dimension.")
    parser.add_argument("--dropout", type=float, default=0.2, help="PyTorch MLP dropout.")
    parser.add_argument(
        "--neural-models",
        default=DEFAULT_NEURAL_MODELS,
        help="Comma-separated neural models: shallow_mlp,pytorch_mlp,deep_mlp,autoencoder_mlp.",
    )
    parser.add_argument(
        "--autoencoder-epochs",
        type=int,
        default=10,
        help="Autoencoder pretraining epochs for autoencoder_mlp.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.generate_sample:
        csv_path = generate_sample_flows(Path("data/sample_flows.csv"), random_state=args.random_state)
    elif args.csv:
        csv_path = args.csv
    else:
        raise SystemExit("Pass --generate-sample or --csv path/to/dataset.csv")

    print(f"Loading data from {csv_path}")
    features, attack_labels = load_flow_csv(csv_path, label_column=args.label_column)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset_profile(features, attack_labels).to_csv(args.output_dir / "dataset_profile.csv", index=False)

    data = prepare_data(
        features,
        attack_labels,
        max_rows=args.max_rows,
        random_state=args.random_state,
    )

    print(f"Rows: train={len(data.y_train)}, val={len(data.y_val)}, test={len(data.y_test)}")
    print(f"Features: {len(data.feature_names)}")
    with (args.artifact_dir / "scaler.pkl").open("wb") as file:
        pickle.dump(data.scaler, file)
    (args.artifact_dir / "feature_names.json").write_text(
        json.dumps(data.feature_names, indent=2),
        encoding="utf-8",
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=args.random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=args.random_state,
        ),
    }

    metric_rows = []
    attack_frames = []
    attack_family_frames = []
    prediction_frame = pd.DataFrame(
        {
            "original_attack_label": data.attack_test.astype(str),
            "binary_true": data.y_test,
        }
    )

    for model_name, model in models.items():
        print(f"\nTraining {model_name}")
        model.fit(data.x_train, data.y_train)
        safe_name = safe_model_name(model_name)
        with (args.artifact_dir / f"{safe_name}.pkl").open("wb") as file:
            pickle.dump(model, file)
        y_pred = model.predict(data.x_test)
        prediction_frame[f"{model_name}_pred"] = y_pred
        print_report(model_name, data.y_test, y_pred)
        metric_rows.append(summarize_binary_metrics(model_name, data.y_test, y_pred))
        attack_frames.append(
            attack_type_detection(model_name, data.attack_test, data.y_test, y_pred)
        )
        attack_family_frames.append(
            attack_family_detection(model_name, data.attack_test, data.y_test, y_pred)
        )
        save_confusion_matrix(model_name, data.y_test, y_pred, args.output_dir)

    neural_model_keys = [model.strip() for model in args.neural_models.split(",") if model.strip()]
    neural_metadata = {}
    for model_key in neural_model_keys:
        print(f"\nTraining neural model: {model_key}")
        neural_model, config = train_neural_model(
            model_key,
            data.x_train,
            data.y_train,
            data.x_val,
            data.y_val,
            epochs=args.epochs,
            autoencoder_epochs=args.autoencoder_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
            random_state=args.random_state,
        )
        model_name = config["display_name"]
        safe_name = safe_model_name(model_name)
        save_mlp_checkpoint(
            neural_model,
            args.artifact_dir / f"{safe_name}.pt",
            input_dim=data.x_train.shape[1],
            config=config,
        )
        y_pred = predict_mlp(neural_model, data.x_test)
        prediction_frame[f"{model_name}_pred"] = y_pred
        print_report(model_name, data.y_test, y_pred)
        metric_rows.append(summarize_binary_metrics(model_name, data.y_test, y_pred))
        attack_frames.append(attack_type_detection(model_name, data.attack_test, data.y_test, y_pred))
        attack_family_frames.append(attack_family_detection(model_name, data.attack_test, data.y_test, y_pred))
        save_confusion_matrix(model_name, data.y_test, y_pred, args.output_dir)
        neural_metadata[model_key] = {
            **config,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "threshold": 0.5,
        }

    metrics = pd.DataFrame(metric_rows)
    attack_results = pd.concat(attack_frames, ignore_index=True)
    attack_family_results = pd.concat(attack_family_frames, ignore_index=True)

    metrics.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    attack_results.to_csv(args.output_dir / "attack_type_detection.csv", index=False)
    attack_family_results.to_csv(args.output_dir / "attack_family_detection.csv", index=False)
    prediction_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)
    save_attack_type_plot(attack_results, args.output_dir)
    save_attack_family_plot(attack_family_results, args.output_dir)
    save_false_negative_plot(metrics, args.output_dir)
    save_model_comparison_findings(attack_results, args.output_dir)

    metadata = {
        "dataset_path": str(csv_path),
        "label_column": args.label_column,
        "max_rows": args.max_rows,
        "random_state": args.random_state,
        "rows": {
            "train": int(len(data.y_train)),
            "validation": int(len(data.y_val)),
            "test": int(len(data.y_test)),
        },
        "feature_count": int(len(data.feature_names)),
        "models": {
            "logistic_regression": models["Logistic Regression"].get_params(),
            "random_forest": models["Random Forest"].get_params(),
            **neural_metadata,
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nSaved reports:")
    print(f"- {args.output_dir / 'metrics_summary.csv'}")
    print(f"- {args.output_dir / 'attack_type_detection.csv'}")
    print(f"- {args.output_dir / 'attack_family_detection.csv'}")
    print(f"- {args.output_dir / 'test_predictions.csv'}")
    print(f"- {args.output_dir / 'attack_type_detection.png'}")
    print(f"- {args.output_dir / 'attack_family_detection.png'}")
    print(f"- {args.output_dir / 'false_negative_rate.png'}")
    print(f"- {args.output_dir / 'model_comparison_findings.md'}")
    print(f"\nSaved artifacts under {args.artifact_dir}")


if __name__ == "__main__":
    main()
