
import csv
import gc
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

class StreamingDataConfig:
    """Configuration for streaming data processing and fairness auditing."""

    def __init__(
        self,
        filepath: str | Path,
        protected_attribute: str,
        privileged_value: str | int,
        target_column: str,
        chunk_size: int = 50000,
        positive_class_value=None,
        prediction_positive_value=None,
    ):
        self.filepath = Path(filepath)
        self.protected_attribute = protected_attribute
        self.privileged_value = privileged_value
        self.target_column = target_column
        self.chunk_size = chunk_size

        # Explicit positive labels are strongly recommended.
        self.positive_class_value = positive_class_value
        self.prediction_positive_value = prediction_positive_value

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if not self.filepath.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {self.filepath.resolve()}"
            )


# ============================================================
# 2. SHARED HELPER FUNCTIONS
# ============================================================

def normalize_value(value):
    """Normalize values for consistent categorical comparisons."""

    if pd.isna(value):
        return None

    return str(value).strip().upper()


def validate_csv_columns(filepath):
    """Detect duplicate column names in the original CSV header."""

    with open(filepath, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        headers = next(reader, None)

    if not headers:
        raise ValueError("The CSV file is empty or has no header.")

    normalized_headers = [str(col).strip() for col in headers]

    duplicates = [
        col for col, count in Counter(normalized_headers).items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"Duplicate column names found in CSV: {duplicates}. "
            "Rename the duplicate columns before processing."
        )

    if any(not col for col in normalized_headers):
        raise ValueError("The CSV contains empty column names.")

    return normalized_headers


def validate_required_columns(columns, required):
    """Check whether all required columns exist."""

    missing = [col for col in required if col not in columns]

    if missing:
        raise ValueError(
            f"Required columns not found in dataset: {missing}. "
            f"Available columns: {list(columns)}"
        )


def resolve_positive_label(
    filepath,
    column,
    chunk_size,
    configured_value=None,
):
    """
    Resolve the positive class.

    Prefer explicitly configured positive_class_value.
    If not provided, use the most frequent class for backward compatibility.
    """

    if configured_value is not None:
        return normalize_value(configured_value)

    warnings.warn(
        f"No positive class configured for '{column}'. "
        "Using the most frequent class as a fallback. "
        "Set positive_class_value explicitly for reliable fairness metrics.",
        UserWarning,
    )

    counts = Counter()

    for chunk in pd.read_csv(
        filepath,
        usecols=[column],
        chunksize=chunk_size,
        low_memory=True,
    ):
        values = chunk[column].dropna().map(normalize_value)
        counts.update(values)

    if not counts:
        raise ValueError(
            f"Column '{column}' contains no valid values."
        )

    return counts.most_common(1)[0][0]


# ============================================================
# 3. OUT-OF-CORE DATA PROFILER
# ============================================================

class OutOfCoreProfiler:
    """
    Computes numerical statistics incrementally.

    Does not load the complete dataset into memory.
    """

    def __init__(self, config: StreamingDataConfig):
        self.config = config

    def compute_global_numerical_metrics(self):

        validate_csv_columns(self.config.filepath)

        # Per-column running statistics
        counts = {}
        means = {}
        M2 = {}

        missing_counts = None
        total_rows = 0

        chunks = pd.read_csv(
            self.config.filepath,
            chunksize=self.config.chunk_size,
            low_memory=True,
        )

        for chunk in chunks:

            total_rows += len(chunk)

            chunk_missing = chunk.isnull().sum()

            if missing_counts is None:
                missing_counts = chunk_missing
            else:
                missing_counts = missing_counts.add(
                    chunk_missing,
                    fill_value=0,
                )

            numeric_df = chunk.select_dtypes(include=[np.number])

            for col in numeric_df.columns:

                values = numeric_df[col].dropna()

                if values.empty:
                    continue

                n2 = len(values)
                mean2 = values.mean()

                M2_2 = ((values - mean2) ** 2).sum()

                if col not in counts:

                    counts[col] = n2
                    means[col] = mean2
                    M2[col] = M2_2

                else:

                    n1 = counts[col]
                    mean1 = means[col]

                    delta = mean2 - mean1
                    total = n1 + n2

                    means[col] = mean1 + delta * n2 / total

                    M2[col] += (
                        M2_2
                        + (delta ** 2) * n1 * n2 / total
                    )

                    counts[col] = total

            del chunk, numeric_df
            gc.collect()

        if not counts:

            return pd.DataFrame(
                columns=[
                    "Total Valid Entries",
                    "Global Mean",
                    "Standard Deviation",
                    "Total Missing",
                ]
            ), total_rows

        std = {
            col: np.sqrt(M2[col] / (counts[col] - 1))
            if counts[col] > 1
            else 0.0
            for col in counts
        }

        summary_df = pd.DataFrame({
            "Total Valid Entries": pd.Series(counts),
            "Global Mean": pd.Series(means),
            "Standard Deviation": pd.Series(std),
            "Total Missing": missing_counts.reindex(
                counts.keys()
            ).fillna(0),
        })

        return summary_df, total_rows


# ============================================================
# 4. LOW-MEMORY PREPROCESSOR
# ============================================================

class LowMemoryPreprocessor:
    """
    Processes large CSV datasets in chunks.

    Operations:
    - Missing value handling
    - Numerical standardization
    - Protected attribute encoding
    - Target label encoding
    """

    def __init__(
        self,
        config: StreamingDataConfig,
        global_stats: pd.DataFrame,
    ):
        self.config = config
        self.global_stats = global_stats

    def process_and_save_stream(self, output_path):

        validate_csv_columns(self.config.filepath)

        out_path = Path(output_path)

        if out_path.resolve() == self.config.filepath.resolve():
            raise ValueError(
                "Output path must be different from input path."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve positive class consistently across chunks.
        target_pos_value = resolve_positive_label(
            self.config.filepath,
            self.config.target_column,
            self.config.chunk_size,
            self.config.positive_class_value,
        )

        chunks = pd.read_csv(
            self.config.filepath,
            chunksize=self.config.chunk_size,
            low_memory=True,
        )

        first_chunk = True

        for chunk in chunks:

            # ------------------------------------------------
            # 1. Handle missing numerical values
            # ------------------------------------------------

            numeric_cols = chunk.select_dtypes(
                include=[np.number]
            ).columns

            for col in numeric_cols:

                if col in self.global_stats.index:

                    mean_value = self.global_stats.loc[
                        col, "Global Mean"
                    ]

                    chunk[col] = chunk[col].fillna(mean_value)

            # ------------------------------------------------
            # 2. Handle missing categorical values
            # ------------------------------------------------

            categorical_cols = chunk.select_dtypes(
                include=["object", "category", "string"]
            ).columns

            for col in categorical_cols:

                chunk[col] = chunk[col].fillna("UNKNOWN_VAL")

            # ------------------------------------------------
            # 3. Standardize numerical columns
            # ------------------------------------------------

            numeric_cols = chunk.select_dtypes(
                include=[np.number]
            ).columns

            for col in numeric_cols:

                if col not in self.global_stats.index:
                    continue

                mean_value = self.global_stats.loc[
                    col, "Global Mean"
                ]

                std_value = self.global_stats.loc[
                    col, "Standard Deviation"
                ]

                if pd.notna(std_value) and std_value > 0:

                    chunk[col] = (
                        chunk[col] - mean_value
                    ) / std_value

            # ------------------------------------------------
            # 4. Encode protected attribute
            # ------------------------------------------------

            protected_col = self.config.protected_attribute

            if protected_col in chunk.columns:

                privileged = normalize_value(
                    self.config.privileged_value
                )

                chunk["is_privileged"] = (
                    chunk[protected_col]
                    .map(normalize_value)
                    .eq(privileged)
                    .astype("int8")
                )

            # ------------------------------------------------
            # 5. Encode target variable
            # ------------------------------------------------

            target_col = self.config.target_column

            if target_col in chunk.columns:

                chunk["target_label"] = (
                    chunk[target_col]
                    .map(normalize_value)
                    .eq(target_pos_value)
                    .astype("int8")
                )

            # ------------------------------------------------
            # 6. Downcast floating-point values
            # ------------------------------------------------

            float_cols = chunk.select_dtypes(
                include=["float64"]
            ).columns

            for col in float_cols:

                chunk[col] = pd.to_numeric(
                    chunk[col],
                    downcast="float",
                )

            # ------------------------------------------------
            # 7. Save processed chunk
            # ------------------------------------------------

            chunk.to_csv(
                out_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )

            first_chunk = False

            del chunk
            gc.collect()

        print(
            f"Preprocessing completed successfully: {out_path.resolve()}"
        )


# ============================================================
# 5. STREAMING FAIRNESS AUDITOR
# ============================================================

class StreamingFairnessAuditor:
    """
    Computes fairness metrics incrementally.

    Metrics:
    1. Demographic Parity Difference
    2. Disparate Impact Ratio
    3. True Positive Rate Difference
    4. False Positive Rate Difference
    """

    def __init__(
        self,
        config: StreamingDataConfig,
        prediction_col: str,
    ):
        self.config = config
        self.prediction_col = prediction_col

    def execute_audit(self):

        validate_csv_columns(self.config.filepath)

        required = [
            self.config.protected_attribute,
            self.config.target_column,
            self.prediction_col,
        ]

        # These three columns must represent distinct variables.
        if len(set(required)) != len(required):
            raise ValueError(
                "Protected attribute, target column, and prediction "
                "column must have different column names."
            )

        validate_required_columns(
            validate_csv_columns(self.config.filepath),
            required,
        )

        # ----------------------------------------------------
        # Resolve positive class values
        # ----------------------------------------------------

        true_pos_value = resolve_positive_label(
            self.config.filepath,
            self.config.target_column,
            self.config.chunk_size,
            self.config.positive_class_value,
        )

        pred_pos_value = (
            normalize_value(self.config.prediction_positive_value)
            if self.config.prediction_positive_value is not None
            else true_pos_value
        )

        privileged_value = normalize_value(
            self.config.privileged_value
        )

        # ----------------------------------------------------
        # Initialize counters
        # ----------------------------------------------------

        counts = {
            "privileged": {
                "total": 0,
                "pos_pred": 0,
                "tpr_num": 0,
                "tpr_den": 0,
                "fpr_num": 0,
                "fpr_den": 0,
            },
            "protected": {
                "total": 0,
                "pos_pred": 0,
                "tpr_num": 0,
                "tpr_den": 0,
                "fpr_num": 0,
                "fpr_den": 0,
            },
        }

        # ----------------------------------------------------
        # Process dataset chunk by chunk
        # ----------------------------------------------------

        chunks = pd.read_csv(
            self.config.filepath,
            usecols=required,
            chunksize=self.config.chunk_size,
            low_memory=True,
        )

        for chunk in chunks:

            # Remove rows with missing required values.
            eval_df = chunk.dropna(
                subset=required
            ).copy()

            if eval_df.empty:
                del chunk, eval_df
                continue

            # ------------------------------------------------
            # FIX: Ensure all selected columns are Series
            # ------------------------------------------------

            y_true_values = (
                eval_df[self.config.target_column]
                .map(normalize_value)
            )

            y_pred_values = (
                eval_df[self.prediction_col]
                .map(normalize_value)
            )

            # Binary actual labels
            y_true = (
                y_true_values == true_pos_value
            ).astype("int8")

            # Binary predicted labels
            y_pred = (
                y_pred_values == pred_pos_value
            ).astype("int8")

            # Protected group identification
            protected_values = (
                eval_df[self.config.protected_attribute]
                .map(normalize_value)
            )

            priv_mask = (
                protected_values == privileged_value
            )

            # ------------------------------------------------
            # Calculate group-level counts
            # ------------------------------------------------

            for group, mask in [
                ("privileged", priv_mask),
                ("protected", ~priv_mask),
            ]:

                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]

                counts[group]["total"] += len(group_y_true)

                counts[group]["pos_pred"] += int(
                    group_y_pred.sum()
                )

                # True Positive Rate
                counts[group]["tpr_num"] += int(
                    (
                        (group_y_true == 1)
                        & (group_y_pred == 1)
                    ).sum()
                )

                counts[group]["tpr_den"] += int(
                    (group_y_true == 1).sum()
                )

                # False Positive Rate
                counts[group]["fpr_num"] += int(
                    (
                        (group_y_true == 0)
                        & (group_y_pred == 1)
                    ).sum()
                )

                counts[group]["fpr_den"] += int(
                    (group_y_true == 0).sum()
                )

            del chunk, eval_df
            gc.collect()

        return self._calculate_final_ratios(counts)

    # ========================================================
    # 6. FINAL FAIRNESS METRICS
    # ========================================================

    def _calculate_final_ratios(self, counts):

        def safe_divide(numerator, denominator):

            if denominator == 0:
                return np.nan

            return numerator / denominator

        # Demographic parity
        p_rate_priv = safe_divide(
            counts["privileged"]["pos_pred"],
            counts["privileged"]["total"],
        )

        p_rate_prot = safe_divide(
            counts["protected"]["pos_pred"],
            counts["protected"]["total"],
        )

        # True positive rate
        tpr_priv = safe_divide(
            counts["privileged"]["tpr_num"],
            counts["privileged"]["tpr_den"],
        )

        tpr_prot = safe_divide(
            counts["protected"]["tpr_num"],
            counts["protected"]["tpr_den"],
        )

        # False positive rate
        fpr_priv = safe_divide(
            counts["privileged"]["fpr_num"],
            counts["privileged"]["fpr_den"],
        )

        fpr_prot = safe_divide(
            counts["protected"]["fpr_num"],
            counts["protected"]["fpr_den"],
        )

        # Disparate impact
        disparate_impact = safe_divide(
            p_rate_prot,
            p_rate_priv,
        )

        return {
            "Demographic Parity Diff": (
                p_rate_prot - p_rate_priv
                if pd.notna(p_rate_prot)
                and pd.notna(p_rate_priv)
                else np.nan
            ),

            "Disparate Impact Ratio": disparate_impact,

            "True Positive Rate Diff": (
                tpr_prot - tpr_priv
                if pd.notna(tpr_prot)
                and pd.notna(tpr_priv)
                else np.nan
            ),

            "False Positive Rate Diff": (
                fpr_prot - fpr_priv
                if pd.notna(fpr_prot)
                and pd.notna(fpr_priv)
                else np.nan
            ),
        }
