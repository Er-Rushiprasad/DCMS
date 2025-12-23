import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "Results/New_PC_With_P1_Filled.csv"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_FILE = f"pc_to_params_model_{TIMESTAMP}.pkl"
SCALER_FILE = f"pc_to_params_scaler_{TIMESTAMP}.pkl"
METADATA_FILE = f"pc_to_params_metadata_{TIMESTAMP}.pkl"


def read_headers(path: str) -> Tuple[List[str], List[str]]:
    with open(path, "r", encoding="latin-1") as fh:
        group_line = fh.readline().strip()
        param_line = fh.readline().strip()
    groups = [g.strip() for g in group_line.split(",")]
    params = [p.strip() for p in param_line.split(",")]
    return groups, params


def parse_group_ranges(groups: List[str]) -> Dict[str, Tuple[int, int]]:
    ranges: Dict[str, Tuple[int, int]] = {}
    current = None
    start = None

    for idx, name in enumerate(groups):
        name_upper = name.upper()
        if name and name_upper not in {"GROUP NO", "GROUP NO.", "GROUP"}:
            if current != name:
                if current is not None and start is not None:
                    ranges[current] = (start, idx - 1)
                current = name
                start = idx
    if current is not None and start is not None:
        ranges[current] = (start, len(groups) - 1)
    return ranges


def build_column_map(params: List[str], ranges: Dict[str, Tuple[int, int]]):
    """Return mappings for PC inputs and P1-P5 outputs."""
    pc_columns: List[Tuple[str, int]] = []
    output_columns: List[Tuple[str, str, int]] = []

    for group, (start, end) in ranges.items():
        for col_idx in range(start, end + 1):
            if col_idx >= len(params):
                continue
            param = params[col_idx].upper()
            if param == "PC":
                pc_columns.append((group, col_idx))
            elif param in {"P1", "P2", "P3", "P4", "P5"}:
                output_columns.append((group, param, col_idx))
            else:
                # ignore P0, PN, PR, PF, etc.
                continue

    # Preserve the order of groups from the header
    group_order = list(dict.fromkeys([g for g, _ in pc_columns]))
    return pc_columns, output_columns, group_order


def to_numeric(value):
    try:
        return pd.to_numeric(value, errors="coerce")
    except Exception:
        return np.nan


def fill_nan_with_median(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    for col in range(arr.shape[1]):
        col_data = arr[:, col]
        mask = np.isnan(col_data)
        if mask.any():
            median = np.nanmedian(col_data)
            if np.isnan(median):
                median = 0.0
            col_data[mask] = median
            arr[:, col] = col_data
    return arr


def load_data() -> Tuple[np.ndarray, np.ndarray, Dict]:
    groups, params = read_headers(INPUT_FILE)
    ranges = parse_group_ranges(groups)
    pc_cols, out_cols, group_order = build_column_map(params, ranges)

    # Load data rows
    df = pd.read_csv(INPUT_FILE, skiprows=2, encoding="latin-1")

    X_rows: List[List[float]] = []
    y_rows: List[List[float]] = []

    for _, row in df.iterrows():
        pc_vals = []
        out_vals = []

        for group, col_idx in pc_cols:
            pc_vals.append(to_numeric(row.iloc[col_idx]))

        for group, param, col_idx in out_cols:
            out_vals.append(to_numeric(row.iloc[col_idx]))

        # Keep only rows with at least one non-NaN input and output
        if (not all(np.isnan(pc_vals))) and (not all(np.isnan(out_vals))):
            X_rows.append(pc_vals)
            y_rows.append(out_vals)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=float)

    X = fill_nan_with_median(X)
    y = fill_nan_with_median(y)

    metadata = {
        "group_order": group_order,
        "pc_columns": pc_cols,
        "output_columns": out_cols,  # list of (group, param, col_idx)
        "pc_feature_count": len(pc_cols),
        "output_feature_count": len(out_cols),
    }

    return X, y, metadata


def train():
    print("Loading data from", INPUT_FILE)
    X, y, metadata = load_data()
    print(f"Input matrix shape: {X.shape}")
    print(f"Output matrix shape: {y.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    print(f"Train R^2: {train_r2:.4f}")
    print(f"Test R^2: {test_r2:.4f}")

    # Save artifacts
    with open(MODEL_FILE, "wb") as fh:
        pickle.dump(model, fh)
    with open(SCALER_FILE, "wb") as fh:
        pickle.dump(scaler, fh)
    with open(METADATA_FILE, "wb") as fh:
        pickle.dump(metadata, fh)

    print("\nArtifacts saved:")
    print("  Model   :", MODEL_FILE)
    print("  Scaler  :", SCALER_FILE)
    print("  Metadata:", METADATA_FILE)
    print("\nMetadata summary:")
    print(f"  Groups: {metadata['group_order']}")
    print(f"  PC features: {metadata['pc_feature_count']}")
    print(f"  Output features: {metadata['output_feature_count']}")


if __name__ == "__main__":
    train()

