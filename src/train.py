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
    attack_type_detection,
    print_report,
    save_attack_type_plot,
    save_confusion_matrix,
    save_false_negative_plot,
    save_model_comparison_findings,
    summarize_binary_metrics,
)
from models import predict_mlp, save_mlp_checkpoint, train_mlp


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
    parser.add_argument("--dropout", type=float, default=0.2, help="PyTorch MLP dropout.")
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
    prediction_frame = pd.DataFrame(
        {
            "original_attack_label": data.attack_test.astype(str),
            "binary_true": data.y_test,
        }
    )

    for model_name, model in models.items():
        print(f"\nTraining {model_name}")
        model.fit(data.x_train, data.y_train)
        safe_name = model_name.lower().replace(" ", "_")
        with (args.artifact_dir / f"{safe_name}.pkl").open("wb") as file:
            pickle.dump(model, file)
        y_pred = model.predict(data.x_test)
        prediction_frame[f"{model_name}_pred"] = y_pred
        print_report(model_name, data.y_test, y_pred)
        metric_rows.append(summarize_binary_metrics(model_name, data.y_test, y_pred))
        attack_frames.append(
            attack_type_detection(model_name, data.attack_test, data.y_test, y_pred)
        )
        save_confusion_matrix(model_name, data.y_test, y_pred, args.output_dir)

    print("\nTraining PyTorch MLP")
    mlp = train_mlp(
        data.x_train,
        data.y_train,
        data.x_val,
        data.y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        random_state=args.random_state,
    )
    save_mlp_checkpoint(
        mlp,
        args.artifact_dir / "pytorch_mlp.pt",
        input_dim=data.x_train.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    mlp_pred = predict_mlp(mlp, data.x_test)
    prediction_frame["PyTorch MLP_pred"] = mlp_pred
    print_report("PyTorch MLP", data.y_test, mlp_pred)
    metric_rows.append(summarize_binary_metrics("PyTorch MLP", data.y_test, mlp_pred))
    attack_frames.append(attack_type_detection("PyTorch MLP", data.attack_test, data.y_test, mlp_pred))
    save_confusion_matrix("PyTorch MLP", data.y_test, mlp_pred, args.output_dir)

    metrics = pd.DataFrame(metric_rows)
    attack_results = pd.concat(attack_frames, ignore_index=True)

    metrics.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    attack_results.to_csv(args.output_dir / "attack_type_detection.csv", index=False)
    prediction_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)
    save_attack_type_plot(attack_results, args.output_dir)
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
            "pytorch_mlp": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "threshold": 0.5,
            },
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nSaved reports:")
    print(f"- {args.output_dir / 'metrics_summary.csv'}")
    print(f"- {args.output_dir / 'attack_type_detection.csv'}")
    print(f"- {args.output_dir / 'test_predictions.csv'}")
    print(f"- {args.output_dir / 'attack_type_detection.png'}")
    print(f"- {args.output_dir / 'false_negative_rate.png'}")
    print(f"- {args.output_dir / 'model_comparison_findings.md'}")
    print(f"\nSaved artifacts under {args.artifact_dir}")


if __name__ == "__main__":
    main()
