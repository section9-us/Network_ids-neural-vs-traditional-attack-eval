from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_LABEL_CANDIDATES = ("Label", "label", "Attack", "attack", "class", "Class")
DEFAULT_BENIGN_NAMES = {"benign", "normal", "0", "false"}


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    attack_train: pd.Series
    attack_val: pd.Series
    attack_test: pd.Series
    feature_names: list[str]
    scaler: StandardScaler


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seen: dict[str, int] = {}
    columns = []
    for col in df.columns:
        normalized = str(col).strip()
        count = seen.get(normalized, 0)
        seen[normalized] = count + 1
        columns.append(normalized if count == 0 else f"{normalized}.{count}")
    df.columns = columns
    return df


def find_label_column(df: pd.DataFrame, label_column: str | None = None) -> str:
    if label_column:
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' was not found.")
        return label_column

    for candidate in DEFAULT_LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not find a label column. Pass --label-column explicitly."
    )


def to_binary_label(labels: pd.Series, benign_names: set[str] | None = None) -> np.ndarray:
    normalized = labels.astype(str).str.strip().str.lower()
    benign_names = benign_names or DEFAULT_BENIGN_NAMES
    return (~normalized.isin(benign_names)).astype(int).to_numpy()


def clean_features(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    labels = df[label_column].astype(str).str.strip()
    features = df.drop(columns=[label_column])

    numeric = features.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    repeated_header_rows = labels.str.lower() == str(label_column).strip().lower()
    if repeated_header_rows.any():
        labels = labels.loc[~repeated_header_rows]
        numeric = numeric.loc[~repeated_header_rows]

    # Drop columns that are entirely missing after conversion.
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.fillna(0.0)

    return numeric, labels


def align_features_for_inference(
    df: pd.DataFrame,
    feature_names: list[str],
    label_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    df = normalize_columns(df)
    labels = None
    if label_column is None:
        try:
            label_column = find_label_column(df)
        except ValueError:
            label_column = None

    if label_column and label_column in df.columns:
        labels = df[label_column].astype(str).str.strip()
        df = df.drop(columns=[label_column])

    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.reindex(columns=feature_names)
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.fillna(0.0)
    return numeric, labels


def load_flow_csv(path: Path, label_column: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    label_col = find_label_column(df, label_column)
    return clean_features(df, label_col)


def dataset_profile(features: pd.DataFrame, attack_labels: pd.Series) -> pd.DataFrame:
    y = to_binary_label(attack_labels)
    rows = [
        {"metric": "rows", "value": int(len(features))},
        {"metric": "features", "value": int(features.shape[1])},
        {"metric": "benign_rows", "value": int((y == 0).sum())},
        {"metric": "attack_rows", "value": int((y == 1).sum())},
    ]
    for attack_label, count in attack_labels.astype(str).value_counts().sort_index().items():
        rows.append({"metric": f"label::{attack_label}", "value": int(count)})
    return pd.DataFrame(rows)


def sample_by_attack_type(
    features: pd.DataFrame,
    attack_labels: pd.Series,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or len(features) <= max_rows:
        return features, attack_labels

    df = features.copy()
    df["_attack_label"] = attack_labels.to_numpy()

    sampled_groups = []
    for _, group in df.groupby("_attack_label"):
        group_size = max(1, int(max_rows * len(group) / len(df)))
        group_size = min(group_size, len(group))
        sampled_groups.append(group.sample(n=group_size, random_state=random_state))

    sampled = pd.concat(sampled_groups, ignore_index=True).sample(
        frac=1.0, random_state=random_state
    )

    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)

    sampled_labels = sampled.pop("_attack_label")
    return sampled.reset_index(drop=True), sampled_labels.reset_index(drop=True)


def prepare_data(
    features: pd.DataFrame,
    attack_labels: pd.Series,
    max_rows: int | None = None,
    random_state: int = 42,
    benign_names: set[str] | None = None,
) -> PreparedData:
    features, attack_labels = sample_by_attack_type(
        features, attack_labels, max_rows=max_rows, random_state=random_state
    )

    y = to_binary_label(attack_labels, benign_names=benign_names)
    stratify = y if len(np.unique(y)) == 2 else None

    x_train_val, x_test, y_train_val, y_test, attack_train_val, attack_test = train_test_split(
        features,
        y,
        attack_labels,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    stratify_train_val = y_train_val if len(np.unique(y_train_val)) == 2 else None
    x_train, x_val, y_train, y_val, attack_train, attack_val = train_test_split(
        x_train_val,
        y_train_val,
        attack_train_val,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return PreparedData(
        x_train=x_train_scaled.astype(np.float32),
        x_val=x_val_scaled.astype(np.float32),
        x_test=x_test_scaled.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        attack_train=attack_train.reset_index(drop=True),
        attack_val=attack_val.reset_index(drop=True),
        attack_test=attack_test.reset_index(drop=True),
        feature_names=list(features.columns),
        scaler=scaler,
    )


def generate_sample_flows(path: Path, rows: int = 5000, random_state: int = 42) -> Path:
    rng = np.random.default_rng(random_state)
    attacks = np.array(["BENIGN", "DDoS", "Brute Force", "Botnet", "Web Attack"])
    probs = np.array([0.62, 0.18, 0.08, 0.06, 0.06])
    labels = rng.choice(attacks, size=rows, p=probs)

    flow_duration = rng.gamma(shape=2.0, scale=800.0, size=rows)
    total_fwd_packets = rng.poisson(18, rows)
    total_bwd_packets = rng.poisson(14, rows)
    total_length_fwd = rng.gamma(3.0, 300.0, rows)
    total_length_bwd = rng.gamma(2.5, 250.0, rows)
    flow_bytes_s = total_length_fwd / np.maximum(flow_duration, 1) * 1000
    flow_packets_s = (total_fwd_packets + total_bwd_packets) / np.maximum(flow_duration, 1) * 1000
    syn_flag_count = rng.poisson(1, rows)
    ack_flag_count = rng.poisson(8, rows)
    psh_flag_count = rng.poisson(2, rows)
    init_win_bytes_fwd = rng.normal(6000, 1000, rows)

    ddos = labels == "DDoS"
    brute = labels == "Brute Force"
    botnet = labels == "Botnet"
    web = labels == "Web Attack"

    flow_packets_s[ddos] *= rng.uniform(4, 8, ddos.sum())
    flow_bytes_s[ddos] *= rng.uniform(3, 6, ddos.sum())
    syn_flag_count[ddos] += rng.poisson(12, ddos.sum())

    brute_duration_boost = rng.uniform(1.5, 3.0, brute.sum())
    flow_duration[brute] *= brute_duration_boost
    syn_flag_count[brute] += rng.poisson(4, brute.sum())
    ack_flag_count[brute] += rng.poisson(2, brute.sum())

    total_fwd_packets[botnet] += rng.poisson(25, botnet.sum())
    total_bwd_packets[botnet] += rng.poisson(3, botnet.sum())
    psh_flag_count[botnet] += rng.poisson(5, botnet.sum())

    total_length_fwd[web] *= rng.uniform(2, 5, web.sum())
    total_length_bwd[web] *= rng.uniform(2, 4, web.sum())
    psh_flag_count[web] += rng.poisson(6, web.sum())

    df = pd.DataFrame(
        {
            "Flow Duration": flow_duration,
            "Total Fwd Packets": total_fwd_packets,
            "Total Backward Packets": total_bwd_packets,
            "Total Length of Fwd Packets": total_length_fwd,
            "Total Length of Bwd Packets": total_length_bwd,
            "Flow Bytes/s": flow_bytes_s,
            "Flow Packets/s": flow_packets_s,
            "SYN Flag Count": syn_flag_count,
            "ACK Flag Count": ack_flag_count,
            "PSH Flag Count": psh_flag_count,
            "Init_Win_bytes_forward": init_win_bytes_fwd,
            "Label": labels,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
