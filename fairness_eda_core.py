"""
Fairness EDA and model-audit engine for the WHO GHED assignment.

The workflow follows AIF360_GHED_Fairness_Analysis.ipynb:
- explicit country/year/protected/target configuration
- country-group holdout to avoid country leakage
- train-only preprocessing
- Logistic Regression and Random Forest baselines
- group fairness and predictive-performance metrics
- Reweighing mitigation and baseline-vs-mitigated comparison
- bootstrap interval for statistical-parity difference

AIF360 is used for Reweighing when installed. A mathematically equivalent
reweighing calculation is provided as a fallback. Fairness metrics are
computed transparently from the held-out predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


@dataclass
class AuditConfig:
    country_col: str
    year_col: str
    protected_col: str
    target_value_col: str
    feature_columns: List[str]
    privileged_value: str
    target_threshold: Optional[float] = None
    indicator_col: Optional[str] = None
    indicator_value: Optional[str] = None
    test_size: float = 0.20
    random_state: int = RANDOM_STATE
    n_estimators: int = 300
    bootstrap_samples: int = 500


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"Duplicate column headers found: {duplicates}. Rename them before auditing.")
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 3),
    }


def dataset_profile(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True),
    }).sort_values(["missing_pct", "unique_values"], ascending=[False, True])
    numeric = df.select_dtypes(include=np.number).describe().T
    categorical_rows = []
    for col in df.select_dtypes(exclude=np.number).columns:
        categorical_rows.append({
            "column": col,
            "unique_values": int(df[col].nunique(dropna=True)),
            "top_value": str(df[col].mode(dropna=True).iloc[0]) if not df[col].mode(dropna=True).empty else None,
            "top_frequency": int(df[col].value_counts(dropna=True).iloc[0]) if not df[col].value_counts(dropna=True).empty else 0,
        })
    return {
        "column_profile": profile,
        "numeric_summary": numeric,
        "categorical_summary": pd.DataFrame(categorical_rows),
    }


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _performance(y_true, y_pred, y_score) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else np.nan,
    }


def _fairness_metrics(frame: pd.DataFrame, y_true, y_pred) -> Dict[str, float]:
    """Group 0 is unprivileged; group 1 is privileged."""
    protected = frame["protected_binary"].to_numpy(dtype=int)
    yt, yp = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)

    def rate(mask, numerator, denominator=None):
        if not mask.any():
            return np.nan
        if denominator is None:
            return float(np.mean(numerator[mask]))
        denom = int(np.sum(denominator[mask]))
        return float(np.sum(numerator[mask]) / denom) if denom else np.nan

    sel_u = rate(protected == 0, yp)
    sel_p = rate(protected == 1, yp)
    tpr_u = rate((protected == 0) & (yt == 1), yp, yt == 1)
    tpr_p = rate((protected == 1) & (yt == 1), yp, yt == 1)
    fpr_u = rate((protected == 0) & (yt == 0), yp, yt == 0)
    fpr_p = rate((protected == 1) & (yt == 0), yp, yt == 0)
    di = sel_u / sel_p if pd.notna(sel_u) and pd.notna(sel_p) and sel_p != 0 else np.nan
    return {
        "SPD": float(sel_u - sel_p) if pd.notna(sel_u) and pd.notna(sel_p) else np.nan,
        "DI": float(di) if pd.notna(di) else np.nan,
        "EOD": float(tpr_u - tpr_p) if pd.notna(tpr_u) and pd.notna(tpr_p) else np.nan,
        "AOD": float(0.5 * ((tpr_u - tpr_p) + (fpr_u - fpr_p)))
               if all(pd.notna(v) for v in [tpr_u, tpr_p, fpr_u, fpr_p]) else np.nan,
    }


def _group_report(frame: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    protected = frame["protected_binary"].to_numpy(dtype=int)
    yt, yp = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    rows = []
    for group, name in [(1, "Privileged"), (0, "Unprivileged")]:
        mask = protected == group
        if not mask.any():
            rows.append({"Group": name, "N": 0, "Positive rate (actual)": np.nan,
                         "Selection rate (predicted)": np.nan, "Accuracy": np.nan,
                         "Precision": np.nan, "Recall / TPR": np.nan, "F1": np.nan})
            continue
        rows.append({
            "Group": name, "N": int(mask.sum()),
            "Positive rate (actual)": float(np.mean(yt[mask])),
            "Selection rate (predicted)": float(np.mean(yp[mask])),
            "Accuracy": float(accuracy_score(yt[mask], yp[mask])),
            "Precision": float(precision_score(yt[mask], yp[mask], zero_division=0)),
            "Recall / TPR": float(recall_score(yt[mask], yp[mask], zero_division=0)),
            "F1": float(f1_score(yt[mask], yp[mask], zero_division=0)),
        })
    return pd.DataFrame(rows)


def _reweighing_weights(frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Use AIF360 Reweighing if available; otherwise its standard cell-weight formula."""
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.algorithms.preprocessing import Reweighing
        aif_df = pd.DataFrame({
            "protected_binary": frame["protected_binary"].to_numpy(dtype=int),
            "label": np.asarray(labels, dtype=int),
        })
        ds = BinaryLabelDataset(
            df=aif_df, label_names=["label"],
            protected_attribute_names=["protected_binary"],
            favorable_label=1, unfavorable_label=0,
        )
        rw = Reweighing(
            unprivileged_groups=[{"protected_binary": 0}],
            privileged_groups=[{"protected_binary": 1}],
        )
        return np.asarray(rw.fit_transform(ds).instance_weights, dtype=float).ravel()
    except ImportError:
        pass

    g = frame["protected_binary"].to_numpy(dtype=int)
    y = np.asarray(labels, dtype=int)
    n = len(y)
    weights = np.ones(n, dtype=float)
    for group in (0, 1):
        for label in (0, 1):
            cell = (g == group) & (y == label)
            n_cell = int(cell.sum())
            n_group = int((g == group).sum())
            n_label = int((y == label).sum())
            if n_cell and n_group and n_label:
                weights[cell] = (n_group * n_label) / (n * n_cell)
    return weights


def _bootstrap_spd(frame: pd.DataFrame, predictions, n_boot: int, seed: int):
    data = pd.DataFrame({
        "protected_binary": frame["protected_binary"].to_numpy(dtype=int),
        "prediction": np.asarray(predictions, dtype=int),
    })
    rng = np.random.default_rng(seed)
    estimates = []
    groups = {g: data[data.protected_binary == g] for g in (0, 1)}
    if any(part.empty for part in groups.values()):
        return {"lower": np.nan, "median": np.nan, "upper": np.nan}
    for _ in range(max(50, int(n_boot))):
        sampled = []
        for part in groups.values():
            sampled.append(part.iloc[rng.integers(0, len(part), len(part))])
        sample = pd.concat(sampled, ignore_index=True)
        rates = sample.groupby("protected_binary").prediction.mean()
        estimates.append(float(rates.loc[0] - rates.loc[1]))
    lo, med, hi = np.quantile(estimates, [0.025, 0.5, 0.975])
    return {"lower": float(lo), "median": float(med), "upper": float(hi)}


def run_audit(df: pd.DataFrame, config: AuditConfig) -> Dict[str, Any]:
    """Run the complete notebook-aligned experiment and return tables/models for Streamlit."""
    validate_dataframe(df)
    required = [config.country_col, config.year_col, config.protected_col, config.target_value_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required configured columns are missing: {missing}")
    if not config.feature_columns:
        raise ValueError("Select at least one defensible predictor feature.")
    missing_features = [c for c in config.feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(f"Selected feature columns are missing: {missing_features}")

    forbidden = {config.country_col, config.protected_col, config.target_value_col,
                 "protected_binary", "label", "prediction"}
    overlap = forbidden.intersection(config.feature_columns)
    if overlap:
        raise ValueError(f"Remove identifier/protected/target/output columns from predictors: {sorted(overlap)}")
    if config.protected_col in config.feature_columns:
        raise ValueError("The protected attribute must be excluded from predictive features.")

    work = df.copy()
    if config.indicator_col:
        if config.indicator_col not in work.columns:
            raise ValueError(f"Indicator column {config.indicator_col!r} was not found.")
        work = work.loc[work[config.indicator_col].astype(str) == str(config.indicator_value)].copy()
    work[config.target_value_col] = pd.to_numeric(work[config.target_value_col], errors="coerce")
    work = work.dropna(subset=required).copy()
    work[config.protected_col] = work[config.protected_col].astype(str)
    groups = sorted(work[config.protected_col].dropna().unique().tolist())
    if config.privileged_value not in groups:
        raise ValueError(f"Privileged value must match an observed group. Available: {groups}")
    if len(groups) < 2:
        raise ValueError("At least two protected-group categories are required.")
    work["protected_binary"] = (work[config.protected_col] == str(config.privileged_value)).astype(int)

    # Split entire countries when repeated observations exist.
    if work[config.country_col].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_size,
                                     random_state=config.random_state)
        train_idx, test_idx = next(splitter.split(work, groups=work[config.country_col]))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(work)), test_size=config.test_size,
            random_state=config.random_state, stratify=work[config.protected_binary]
        )
    train_raw, test_raw = work.iloc[train_idx].copy(), work.iloc[test_idx].copy()
    if train_raw.empty or test_raw.empty:
        raise ValueError("Train/test split produced an empty partition.")

    threshold = config.target_threshold
    if threshold is None:
        threshold = float(train_raw[config.target_value_col].median())
    train_raw["label"] = (train_raw[config.target_value_col] >= threshold).astype(int)
    test_raw["label"] = (test_raw[config.target_value_col] >= threshold).astype(int)
    if train_raw["label"].nunique() < 2:
        raise ValueError("Training target contains only one class. Adjust the target threshold or data selection.")
    if test_raw["label"].nunique() < 2:
        warnings.warn("Test target contains one class; ROC-AUC will be undefined.")

    X_train_raw = train_raw[config.feature_columns].copy()
    X_test_raw = test_raw[config.feature_columns].copy()
    y_train = train_raw["label"].to_numpy(dtype=int)
    y_test = test_raw["label"].to_numpy(dtype=int)
    numeric_cols = X_train_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in config.feature_columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]), categorical_cols))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=config.random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=config.n_estimators, min_samples_leaf=2,
            random_state=config.random_state, n_jobs=-1,
        ),
    }
    fitted_models, predictions, scores = {}, {}, {}
    performance_rows, fairness_rows = [], []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test).astype(int)
        score = model.predict_proba(X_test)[:, 1]
        fitted_models[name] = model
        predictions[name], scores[name] = pred, score
        performance_rows.append({"Model": name, **_performance(y_test, pred, score)})
        fairness_rows.append({"Model": name, **_fairness_metrics(test_raw, y_test, pred)})

    performance_df = pd.DataFrame(performance_rows).set_index("Model")
    baseline_fairness_df = pd.DataFrame(fairness_rows).set_index("Model")
    selected_name = "Random Forest"
    baseline_model = fitted_models[selected_name]
    baseline_pred = predictions[selected_name]
    sample_weights = _reweighing_weights(train_raw, y_train)
    mitigated_model = RandomForestClassifier(
        n_estimators=config.n_estimators, min_samples_leaf=2,
        random_state=config.random_state, n_jobs=-1,
    )
    mitigated_model.fit(X_train, y_train, sample_weight=sample_weights)
    mitigated_pred = mitigated_model.predict(X_test).astype(int)
    mitigated_score = mitigated_model.predict_proba(X_test)[:, 1]

    baseline_perf = _performance(y_test, baseline_pred, scores[selected_name])
    mitigated_perf = _performance(y_test, mitigated_pred, mitigated_score)
    baseline_fair = _fairness_metrics(test_raw, y_test, baseline_pred)
    mitigated_fair = _fairness_metrics(test_raw, y_test, mitigated_pred)
    comparison = pd.DataFrame([
        {"Stage": "Baseline RF", **baseline_perf, **baseline_fair},
        {"Stage": "Reweighed RF", **mitigated_perf, **mitigated_fair},
    ]).set_index("Stage")
    group_baseline = _group_report(test_raw, y_test, baseline_pred)
    group_mitigated = _group_report(test_raw, y_test, mitigated_pred)
    bootstrap = {
        "Baseline RF": _bootstrap_spd(test_raw, baseline_pred, config.bootstrap_samples, config.random_state),
        "Reweighed RF": _bootstrap_spd(test_raw, mitigated_pred, config.bootstrap_samples, config.random_state + 1),
    }

    # EDA outputs use the filtered analysis population, not just the test subset.
    return {
        "data": work, "train": train_raw, "test": test_raw,
        "threshold": threshold, "preprocessor": preprocessor,
        "models": fitted_models, "mitigated_model": mitigated_model,
        "predictions": predictions, "scores": scores,
        "y_test": y_test, "X_test": X_test,
        "performance": performance_df, "baseline_fairness": baseline_fairness_df,
        "comparison": comparison, "group_baseline": group_baseline,
        "group_mitigated": group_mitigated, "bootstrap": bootstrap,
        "sample_weights": sample_weights,
        "group_counts": work[config.protected_col].value_counts().rename_axis("Group").reset_index(name="Count"),
        "year_counts": work.groupby(config.year_col).size().rename("Count").reset_index(),
        "target_by_group": work.groupby(config.protected_col)[config.target_value_col].agg(["count", "mean", "median"]).reset_index(),
        "config": config,
        "split_summary": {
            "train_rows": int(len(train_raw)), "test_rows": int(len(test_raw)),
            "train_countries": int(train_raw[config.country_col].nunique()),
            "test_countries": int(test_raw[config.country_col].nunique()),
            "threshold": float(threshold),
        },
    }


def export_audit_tables(results: Dict[str, Any]) -> Dict[str, bytes]:
    tables = {
        "baseline_performance.csv": results["performance"],
        "baseline_fairness.csv": results["baseline_fairness"],
        "baseline_vs_mitigated.csv": results["comparison"],
        "group_performance_baseline.csv": results["group_baseline"],
        "group_performance_reweighed.csv": results["group_mitigated"],
        "group_counts.csv": results["group_counts"],
        "target_by_group.csv": results["target_by_group"],
    }
    return {name: frame.to_csv(index=True).encode("utf-8") for name, frame in tables.items()}
