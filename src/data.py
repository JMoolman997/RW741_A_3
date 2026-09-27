from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series
    class_names: list[str]
    source: str
    uci_id: int


@dataclass
class NestedFold:
    dataset: str
    outer_fold: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    scaler: StandardScaler
    class_names: list[str]


def _deduplicate(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True).rename("Class")], axis=1)
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame.drop(columns="Class"), frame["Class"].astype(str)


def _cached_or_fetch(data_dir: Path, filename: str, uci_id: int) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    if path.exists():
        return pd.read_csv(path)
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=uci_id)
    except Exception as exc:
        raise RuntimeError(
            f"UCI dataset {uci_id} is not cached. Connect once or place {filename} in {data_dir}."
        ) from exc
    frame = pd.concat([dataset.data.features.reset_index(drop=True),
                       dataset.data.targets.reset_index(drop=True)], axis=1)
    frame.to_csv(path, index=False)
    return frame


def load_dry_bean(data_dir: str | Path = "data") -> DatasetBundle:
    data_dir = Path(data_dir)
    csv_path, xlsx_path = data_dir / "Dry_Bean_Dataset.csv", data_dir / "Dry_Bean_Dataset.xlsx"
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
    elif xlsx_path.exists():
        frame = pd.read_excel(xlsx_path)
    else:
        frame = _cached_or_fetch(data_dir, csv_path.name, 602)
    target = "Class" if "Class" in frame else frame.columns[-1]
    X, y = _deduplicate(frame.drop(columns=target), frame[target].astype(str))
    return DatasetBundle("dry_bean", X, y, sorted(y.unique()), "UCI Dry Bean Dataset", 602)


def decode_one_hot_targets(targets: pd.DataFrame) -> pd.Series:
    numeric = targets.apply(pd.to_numeric, errors="raise")
    active = numeric.to_numpy() == 1
    if not np.all(active.sum(axis=1) == 1):
        bad = np.flatnonzero(active.sum(axis=1) != 1)[:5].tolist()
        raise ValueError(f"Targets require exactly one active class per row; bad rows: {bad}")
    return pd.Series(numeric.columns[active.argmax(axis=1)], name="Class", dtype=str)


def load_steel_plates(data_dir: str | Path = "data") -> DatasetBundle:
    frame = _cached_or_fetch(Path(data_dir), "Steel_Plates_Faults.csv", 198)
    target_names = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    if "Class" in frame:
        X, y = frame.drop(columns="Class"), frame["Class"].astype(str)
    elif set(target_names).issubset(frame.columns):
        X, y = frame.drop(columns=target_names), decode_one_hot_targets(frame[target_names])
    else:
        X, y = frame.iloc[:, :-7], decode_one_hot_targets(frame.iloc[:, -7:])
    X, y = _deduplicate(X, y)
    if X.shape[1] != 27 or y.nunique() != 7:
        raise ValueError(f"Unexpected Steel Plates data: {X.shape}, {y.nunique()} classes")
    return DatasetBundle("steel_plates", X, y, sorted(y.unique()), "UCI Steel Plates Faults", 198)


def inspect_dataset(bundle: DatasetBundle) -> dict:
    X, y = bundle.X, bundle.y
    counts = y.value_counts().sort_values()
    numeric = X.apply(pd.to_numeric, errors="coerce")
    return {
        "dataset": bundle.name, "instances": len(X), "features": X.shape[1], "classes": y.nunique(),
        "class_counts": counts, "class_proportions": counts / len(y),
        "imbalance_ratio": counts.max() / counts.min(),
        "missing_values": int(X.isna().sum().sum() + y.isna().sum()),
        "invalid_numeric_values": int(numeric.isna().sum().sum() - X.isna().sum().sum()),
        "duplicate_rows": int(pd.concat([X, y.rename("Class")], axis=1).duplicated().sum()),
    }


def make_nested_folds(bundle: DatasetBundle, n_splits: int = 5, seed: int = 42) -> list[NestedFold]:
    mapping = {name: i for i, name in enumerate(bundle.class_names)}
    y = bundle.y.map(mapping).to_numpy(dtype=int)
    X = bundle.X.to_numpy(dtype=float)
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_id, (outer_train, test_idx) in enumerate(outer.split(X, y), start=1):
        train_idx, val_idx = train_test_split(
            outer_train, test_size=.1875, stratify=y[outer_train], random_state=seed + fold_id
        )
        scaler = StandardScaler().fit(X[train_idx])
        folds.append(NestedFold(
            bundle.name, fold_id, scaler.transform(X[train_idx]), scaler.transform(X[val_idx]),
            scaler.transform(X[test_idx]), y[train_idx], y[val_idx], y[test_idx],
            train_idx, val_idx, test_idx, scaler, bundle.class_names,
        ))
    return folds


def class_order(y_train: np.ndarray) -> list[int]:
    labels, counts = np.unique(y_train, return_counts=True)
    return labels[np.lexsort((labels, counts))].tolist()
