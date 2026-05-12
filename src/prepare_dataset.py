from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from data import (
    clean_features,
    dataset_profile,
    find_label_column,
    normalize_columns,
    sample_by_attack_type,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge, clean, profile, and sample CIC-style flow CSV files."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input CSV paths or glob patterns, for example data/raw/*.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ids_sample.csv"),
        help="Processed CSV path for training.",
    )
    parser.add_argument("--label-column", default=None, help="Optional label column name.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50000,
        help="Optional attack-label-aware sample size. Use 0 to keep all rows.",
    )
    parser.add_argument(
        "--min-rows-per-label",
        type=int,
        default=100,
        help="Minimum sampled rows to preserve per label when available.",
    )
    parser.add_argument("--profile-output", type=Path, default=Path("reports/dataset_profile.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("reports/dataset_metadata.json"))
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Rows per CSV chunk while scanning large datasets.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if matches:
            paths.extend(matches)
            continue

        path = Path(pattern)
        if path.exists():
            paths.append(path)

    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise SystemExit("No input CSV files matched --input.")
    return unique_paths


def trim_candidates(
    candidates: list[pd.DataFrame],
    candidate_budget: int,
    min_rows_per_label: int,
    random_state: int,
) -> list[pd.DataFrame]:
    candidate_frame = pd.concat(candidates, ignore_index=True, sort=False)
    labels = candidate_frame.pop("Label").astype(str).str.strip()
    sampled_features, sampled_labels = sample_with_minimum_per_label(
        candidate_frame,
        labels,
        max_rows=candidate_budget,
        min_rows_per_label=min_rows_per_label,
        random_state=random_state,
    )
    sampled = sampled_features.copy()
    sampled["Label"] = sampled_labels.to_numpy()
    return [sampled]


def scan_csv_candidates(
    paths: list[Path],
    label_column: str | None,
    max_rows: int | None,
    min_rows_per_label: int,
    chunksize: int,
    random_state: int,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    candidate_budget = max(max_rows * 4, max_rows + 1000) if max_rows else 0
    candidates: list[pd.DataFrame] = []
    candidate_rows = 0
    total_rows = 0
    label_counts: dict[str, int] = {}

    for path in paths:
        print(f"Reading {path}")
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk = normalize_columns(chunk)
            detected_label_column = find_label_column(chunk, label_column)
            features, labels = clean_features(chunk, detected_label_column)
            total_rows += len(labels)

            for label, count in labels.astype(str).value_counts().items():
                label_counts[label] = label_counts.get(label, 0) + int(count)

            frame = features.copy()
            frame["Label"] = labels.to_numpy()

            if max_rows is None:
                candidates.append(frame)
                continue

            candidates.append(frame)
            candidate_rows += len(frame)
            if candidate_rows > candidate_budget:
                candidates = trim_candidates(
                    candidates,
                    candidate_budget,
                    min_rows_per_label,
                    random_state,
                )
                candidate_rows = len(candidates[0])

    if not candidates:
        raise SystemExit("No rows were read from the input CSV files.")

    return pd.concat(candidates, ignore_index=True, sort=False), total_rows, label_counts


def sample_with_minimum_per_label(
    features: pd.DataFrame,
    labels: pd.Series,
    max_rows: int | None,
    min_rows_per_label: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or len(features) <= max_rows:
        return features, labels

    frame = features.copy()
    frame["Label"] = labels.to_numpy()
    grouped = list(frame.groupby("Label", sort=True))

    reserved = []
    remaining_groups = []
    for _, group in grouped:
        reserve_size = min(len(group), min_rows_per_label)
        reserved.append(group.sample(n=reserve_size, random_state=random_state))
        if len(group) > reserve_size:
            remaining_groups.append(group.drop(reserved[-1].index))

    reserved_frame = pd.concat(reserved, ignore_index=True)
    remaining_budget = max_rows - len(reserved_frame)
    if remaining_budget <= 0:
        sampled = reserved_frame.sample(n=max_rows, random_state=random_state)
    elif remaining_groups:
        remaining_frame = pd.concat(remaining_groups, ignore_index=True)
        extra_features = remaining_frame.drop(columns=["Label"])
        extra_labels = remaining_frame["Label"]
        sampled_extra_features, sampled_extra_labels = sample_by_attack_type(
            extra_features,
            extra_labels,
            max_rows=remaining_budget,
            random_state=random_state,
        )
        sampled_extra = sampled_extra_features.copy()
        sampled_extra["Label"] = sampled_extra_labels.to_numpy()
        sampled = pd.concat([reserved_frame, sampled_extra], ignore_index=True)
    else:
        sampled = reserved_frame

    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sampled_labels = sampled.pop("Label").astype(str).str.strip()
    return sampled, sampled_labels


def main() -> None:
    args = parse_args()
    input_paths = expand_inputs(args.input)
    max_rows = None if args.max_rows == 0 else args.max_rows

    merged, rows_before_sampling, full_label_counts = scan_csv_candidates(
        input_paths,
        label_column=args.label_column,
        max_rows=max_rows,
        min_rows_per_label=args.min_rows_per_label,
        chunksize=args.chunksize,
        random_state=args.random_state,
    )
    labels = merged.pop("Label").astype(str).str.strip()
    features = merged.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

    sampled_features, sampled_labels = sample_with_minimum_per_label(
        features,
        labels,
        max_rows=max_rows,
        min_rows_per_label=args.min_rows_per_label,
        random_state=args.random_state,
    )
    output = sampled_features.copy()
    output["Label"] = sampled_labels.to_numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_profile(sampled_features, sampled_labels).to_csv(args.profile_output, index=False)

    metadata = {
        "input_files": [str(path) for path in input_paths],
        "output": str(args.output),
        "profile_output": str(args.profile_output),
        "rows_before_sampling": int(rows_before_sampling),
        "rows_after_sampling": int(len(output)),
        "feature_count": int(sampled_features.shape[1]),
        "full_label_counts": full_label_counts,
        "label_column_written": "Label",
        "max_rows": max_rows,
        "min_rows_per_label": args.min_rows_per_label,
        "chunksize": args.chunksize,
        "random_state": args.random_state,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved processed dataset to {args.output}")
    print(f"Saved profile to {args.profile_output}")
    print(f"Saved metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
