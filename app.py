"""
Streamlit dashboard for the WHO GHED AI fairness assignment.
Run: streamlit run app.py
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fairness_eda_core import (
    AuditConfig, dataset_profile, export_audit_tables, run_audit, validate_dataframe,
)

st.set_page_config(
    page_title="GHED AI Fairness Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {background: rgba(120,120,120,.07); padding: 12px; border-radius: 10px;}
    .small-note {font-size: .88rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ AI Fairness Audit Dashboard")
st.caption("WHO Global Health Expenditure Data (GHED) | IBM AI Fairness 360-aligned workflow")

st.info(
    "Research scope: GHED contains aggregate country-year observations. "
    "This dashboard assesses country-level group disparities; it does not measure "
    "discrimination against individual patients."
)

with st.sidebar:
    st.header("1. Dataset")
    uploaded = st.file_uploader("Upload GHED CSV", type=["csv"])
    sample_path = st.text_input("Or local CSV path", value="data/GHED_data.csv")
    st.divider()
    st.header("2. Audit settings")
    st.caption("Load a dataset to configure the study.")

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

df = None
if uploaded is not None:
    try:
        df = load_csv(uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
elif sample_path and Path(sample_path).is_file():
    try:
        df = pd.read_csv(sample_path, low_memory=False)
    except Exception as exc:
        st.error(f"Could not read {sample_path}: {exc}")

if df is None:
    st.markdown("### Get started")
    st.write(
        "Upload the WHO GHED CSV to begin. The assignment notebook intentionally "
        "does not include fabricated results or hard-coded column names."
    )
    st.markdown("""
    **Notebook-aligned workflow**
    1. Inspect the original CSV and identify its country, year, indicator, value and group columns.
    2. Filter to one comparable indicator when the file is in long format.
    3. Select defensible predictors and define a documented binary outcome threshold.
    4. Hold out countries, fit preprocessing on training data, and evaluate two baseline models.
    5. Apply Reweighing to training data and compare against the same untouched test set.
    """)
    st.stop()

try:
    data_info = validate_dataframe(df)
    profiles = dataset_profile(df)
except Exception as exc:
    st.error(f"Dataset validation failed: {exc}")
    st.stop()

st.success(f"Loaded {data_info['rows']:,} rows × {data_info['columns']:,} columns.")
overview_tabs = st.tabs(["Dataset overview", "Missing data", "Distributions", "Data preview"])
with overview_tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{data_info['rows']:,}")
    c2.metric("Columns", f"{data_info['columns']:,}")
    c3.metric("Duplicate rows", f"{data_info['duplicate_rows']:,}")
    c4.metric("Memory", f"{data_info['memory_mb']} MB")
    st.subheader("Column profile")
    st.dataframe(profiles["column_profile"], use_container_width=True)
    if not profiles["numeric_summary"].empty:
        st.subheader("Numeric summary")
        st.dataframe(profiles["numeric_summary"], use_container_width=True)
    if not profiles["categorical_summary"].empty:
        st.subheader("Categorical summary")
        st.dataframe(profiles["categorical_summary"], use_container_width=True)
with overview_tabs[1]:
    missing = profiles["column_profile"].query("missing_count > 0").reset_index(names="Column")
    if missing.empty:
        st.success("No missing values detected.")
    else:
        fig = px.bar(missing.sort_values("missing_pct"), x="missing_pct", y="Column",
                     orientation="h", text="missing_pct", title="Missing values by column (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(missing, use_container_width=True)
with overview_tabs[2]:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_numeric = st.selectbox("Numeric variable", numeric_cols) if numeric_cols else None
    if selected_numeric:
        fig = px.histogram(df, x=selected_numeric, marginal="box", title=f"Distribution of {selected_numeric}")
        st.plotly_chart(fig, use_container_width=True)
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    selected_cat = st.selectbox("Categorical variable", categorical_cols, key="eda_cat") if categorical_cols else None
    if selected_cat:
        counts = df[selected_cat].astype(str).value_counts().head(30).rename_axis(selected_cat).reset_index(name="Count")
        st.plotly_chart(px.bar(counts, x=selected_cat, y="Count", title=f"Top categories: {selected_cat}"),
                        use_container_width=True)
with overview_tabs[3]:
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button("Download source preview (CSV)", df.head(100).to_csv(index=False).encode(),
                       "dataset_preview.csv", "text/csv")

st.divider()
st.header("Configure the research experiment")
columns = df.columns.tolist()
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

with st.form("audit_configuration"):
    col1, col2 = st.columns(2)
    country_col = col1.selectbox("Country identifier", columns,
                                 index=columns.index("Country") if "Country" in columns else 0)
    year_col = col2.selectbox("Year column", columns,
                              index=columns.index("Year") if "Year" in columns else min(1, len(columns)-1))
    protected_options = [c for c in columns if c not in [country_col, year_col]]
    protected_col = st.selectbox("Protected/group attribute (country-level)", protected_options,
                                  index=protected_options.index("IncomeGroup") if "IncomeGroup" in protected_options else 0)
    value_options = [c for c in numeric_columns if c not in [country_col, year_col]]
    if not value_options:
        st.error("A numeric target value column is required.")
        st.stop()
    target_col = st.selectbox("Numeric target indicator value", value_options,
                              index=value_options.index("IndicatorValue") if "IndicatorValue" in value_options else 0)
    indicator_options = ["(already filtered / not applicable)"] + [c for c in columns if c not in [country_col, year_col, protected_col, target_col]]
    indicator_choice = st.selectbox("Optional indicator column (long-form data)", indicator_options)
    indicator_value = None
    indicator_col = None
    if indicator_choice != "(already filtered / not applicable)":
        indicator_col = indicator_choice
        choices = sorted(df[indicator_col].dropna().astype(str).unique().tolist())
        indicator_value = st.selectbox("Indicator value to retain", choices)
    group_values = sorted(df[protected_col].dropna().astype(str).unique().tolist())
    privileged_value = st.selectbox("Privileged category (must be domain-justified)", group_values)
    threshold_mode = st.radio("Binary outcome threshold", ["Training-set median", "Specify threshold"], horizontal=True)
    threshold_value = None
    if threshold_mode == "Specify threshold":
        default_threshold = float(pd.to_numeric(df[target_col], errors="coerce").median())
        threshold_value = st.number_input("Favorable outcome threshold", value=default_threshold)
    candidate_features = [c for c in columns if c not in {
        country_col, protected_col, target_col, year_col, "protected_binary", "label", "prediction"
    }]
    default_features = [c for c in candidate_features if c != indicator_col][:min(5, len(candidate_features))]
    features = st.multiselect("Predictor features (exclude identifiers, protected attribute, target and leakage)",
                              candidate_features, default=default_features)
    test_size = st.slider("Test fraction (held-out countries)", 0.10, 0.40, 0.20, 0.05)
    n_estimators = st.slider("Random Forest trees", 100, 600, 300, 50)
    submitted = st.form_submit_button("Run complete fairness audit", type="primary", use_container_width=True)

if submitted:
    try:
        cfg = AuditConfig(
            country_col=country_col, year_col=year_col, protected_col=protected_col,
            target_value_col=target_col, feature_columns=features,
            privileged_value=privileged_value, target_threshold=threshold_value,
            indicator_col=indicator_col, indicator_value=indicator_value,
            test_size=test_size, n_estimators=n_estimators,
        )
        with st.spinner("Preprocessing, training models, calculating fairness and mitigation results..."):
            results = run_audit(df, cfg)
        st.session_state["audit_results"] = results
        st.session_state["audit_config"] = cfg
    except Exception as exc:
        st.exception(exc)

results = st.session_state.get("audit_results")
if results is None:
    st.caption("Configure the study above and select **Run complete fairness audit** to generate model and fairness results.")
    st.stop()

st.divider()
st.header("Analysis results")
summary = results["split_summary"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Analysis observations", f"{len(results['data']):,}")
m2.metric("Training rows", f"{summary['train_rows']:,}")
m3.metric("Test rows", f"{summary['test_rows']:,}")
m4.metric("Outcome threshold", f"{summary['threshold']:,.4g}")
st.caption(f"Country split: {summary['train_countries']} train countries / {summary['test_countries']} test countries.")

tabs = st.tabs([
    "Group & target EDA", "Model performance", "Fairness metrics",
    "Group-level results", "Mitigation comparison", "Uncertainty & exports",
])

with tabs[0]:
    result_cfg = results["config"]
    result_protected_col = result_cfg.protected_col
    result_target_col = result_cfg.target_value_col
    result_year_col = result_cfg.year_col
    left, right = st.columns(2)
    with left:
        st.subheader("Protected-group representation")
        st.plotly_chart(px.bar(results["group_counts"], x="Group", y="Count",
                               title="Country-year observations by group"), use_container_width=True)
    with right:
        st.subheader("Target indicator by protected group")
        long_target = results["data"][[result_protected_col, result_target_col]].copy()
        long_target[result_target_col] = pd.to_numeric(long_target[result_target_col], errors="coerce")
        st.plotly_chart(px.box(long_target, x=result_protected_col, y=result_target_col, points="outliers",
                               title="Target distribution across groups"), use_container_width=True)
    st.subheader("Observations over time")
    st.plotly_chart(px.line(results["year_counts"], x=result_year_col, y="Count", markers=True,
                            title="Country-year coverage by year"), use_container_width=True)
    st.subheader("Mean and median target by group")
    st.dataframe(results["target_by_group"], use_container_width=True)
    st.subheader("Actual outcome prevalence in held-out test data")
    test_view = results["test"].copy()
    test_view["Actual outcome"] = test_view["label"].map({0: "Unfavorable (0)", 1: "Favorable (1)"})
    actual_counts = test_view.groupby([result_protected_col, "Actual outcome"]).size().reset_index(name="Count")
    st.plotly_chart(px.bar(actual_counts, x=result_protected_col, y="Count", color="Actual outcome",
                           barmode="group", title="Actual outcome counts by group"), use_container_width=True)

with tabs[1]:
    st.subheader("Baseline predictive performance")
    perf = results["performance"].reset_index()
    st.dataframe(results["performance"].style.format("{:.3f}"), use_container_width=True)
    perf_long = perf.melt(id_vars="Model", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(perf_long, x="Metric", y="Score", color="Model", barmode="group",
                           title="Logistic Regression vs Random Forest"), use_container_width=True)
    selected_model = st.selectbox("Confusion matrix model", list(results["models"].keys()))
    y_true = results["y_test"]
    y_pred = results["predictions"][selected_model]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = px.imshow(cm, text_auto=True, x=["Predicted 0", "Predicted 1"],
                    y=["Actual 0", "Actual 1"], color_continuous_scale="Blues",
                    title=f"Confusion matrix: {selected_model}")
    st.plotly_chart(fig, use_container_width=True)
    if len(np.unique(y_true)) == 2:
        from sklearn.metrics import roc_curve
        fig = go.Figure()
        for model_name, score in results["scores"].items():
            fpr, tpr, _ = roc_curve(y_true, score)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=model_name))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash"}))
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("ROC curves are unavailable because the held-out test set contains one target class.")

with tabs[2]:
    st.subheader("Baseline fairness metrics (held-out predictions)")
    st.dataframe(results["baseline_fairness"].style.format("{:.4f}"), use_container_width=True)
    fair = results["baseline_fairness"].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Value")
    st.plotly_chart(px.bar(fair, x="Metric", y="Value", color="Model", barmode="group",
                           title="Baseline fairness metrics"), use_container_width=True)
    st.markdown("""
    **Interpretation**
    - **SPD** = unprivileged selection rate − privileged selection rate. Ideal: 0.
    - **DI** = unprivileged selection rate ÷ privileged selection rate. Ideal: 1.
    - **EOD** = unprivileged true-positive rate − privileged true-positive rate. Ideal: 0.
    - **AOD** = average of group TPR and FPR differences. Ideal: 0.
    """)
    st.caption("DI is undefined when the privileged selection rate is zero. Difference metrics are signed; inspect their direction.")

with tabs[3]:
    st.subheader("Group-wise model performance")
    group_metric_format = {
        "Positive rate (actual)": "{:.3f}", "Selection rate (predicted)": "{:.3f}",
        "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall / TPR": "{:.3f}", "F1": "{:.3f}",
    }
    model_to_view = st.selectbox("Model for group report", list(results["models"].keys()), key="group_model")
    if model_to_view == "Random Forest":
        group_df = results["group_baseline"]
    else:
        # Generate group-wise report for selected baseline model from stored predictions.
        from fairness_eda_core import _group_report
        group_df = _group_report(results["test"], results["y_test"], results["predictions"][model_to_view])
    st.dataframe(group_df.style.format(group_metric_format), use_container_width=True)
    group_long = group_df.melt(id_vars=["Group", "N"], var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(group_long, x="Metric", y="Score", color="Group", barmode="group",
                           title="Performance by protected group"), use_container_width=True)
    st.caption("Always interpret rates together with each group's sample size. Small group/class counts make estimates unstable.")

with tabs[4]:
    st.subheader("Reweighing mitigation")
    st.write("Reweighing adjusts training-example weights using the joint distribution of protected group and label. The held-out test data is unchanged.")
    comparison = results["comparison"]
    st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
    perf_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fair_cols = ["SPD", "EOD", "AOD"]
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(comparison[perf_cols].reset_index().melt(id_vars="Stage"),
                               x="variable", y="value", color="Stage", barmode="group",
                               title="Performance before and after Reweighing"), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(comparison[fair_cols].reset_index().melt(id_vars="Stage"),
                               x="variable", y="value", color="Stage", barmode="group",
                               title="Difference metrics (ideal = 0)"), use_container_width=True)
    di_fig = px.bar(comparison[["DI"]].reset_index(), x="Stage", y="DI",
                    title="Disparate impact (ideal = 1)")
    di_fig.add_hline(y=1, line_dash="dash")
    st.plotly_chart(di_fig, use_container_width=True)
    st.subheader("Group performance after Reweighing")
    st.dataframe(results["group_mitigated"].style.format(group_metric_format), use_container_width=True)

with tabs[5]:
    st.subheader("Bootstrap uncertainty for SPD")
    ci_rows = []
    for stage, values in results["bootstrap"].items():
        ci_rows.append({"Stage": stage, "Lower 95%": values["lower"],
                        "Median": values["median"], "Upper 95%": values["upper"]})
    ci_df = pd.DataFrame(ci_rows)
    st.dataframe(ci_df.style.format({
        "Lower 95%": "{:.4f}", "Median": "{:.4f}", "Upper 95%": "{:.4f}",
    }), use_container_width=True)
    fig = go.Figure()
    for _, row in ci_df.iterrows():
        if pd.notna(row["Median"]):
            fig.add_trace(go.Scatter(
                x=[row["Median"]], y=[row["Stage"]], mode="markers",
                error_x={"type": "data", "symmetric": False,
                         "array": [row["Upper 95%"] - row["Median"]],
                         "arrayminus": [row["Median"] - row["Lower 95%"]]},
                name=row["Stage"],
            ))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(title="95% bootstrap interval for SPD", xaxis_title="SPD (ideal = 0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The notebook's demonstration bootstrap resamples observations within each group. For repeated country-year data, a country-cluster bootstrap is preferable.")
    st.subheader("Download audit outputs")
    csv_exports = export_audit_tables(results)
    for filename, payload in csv_exports.items():
        st.download_button(f"Download {filename}", payload, filename, "text/csv", key=filename)
    report_buffer = io.BytesIO()
    with zipfile.ZipFile(report_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, payload in csv_exports.items():
            archive.writestr(filename, payload)
    st.download_button("Download all result tables (ZIP)", report_buffer.getvalue(),
                       "fairness_audit_tables.zip", "application/zip", type="primary")
