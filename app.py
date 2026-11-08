from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from fairness_eda_core import (
    LowMemoryPreprocessor,
    OutOfCoreProfiler,
    StreamingDataConfig,
    StreamingFairnessAuditor,
)

# Set up page configurations cleanly
st.set_page_config(
    page_title="Algorithmic Fairness & Out-of-Core EDA Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard Branding Headers
st.title("⚖️ Socio-Technical Fairness & Large-Scale EDA Application")
st.markdown(
    """
    This portal processes highly scaled datasets safely without overloading system RAM. 
    It computes exact incremental statistics and audits critical fairness risk vectors.
    """
)

# Sidebar Resource & Memory Pipeline Control Setup
st.sidebar.header("📁 Pipeline Resource Configuration")
csv_file_path = st.sidebar.text_input("CSV Target Path", value="data/GHED_data.csv")
chunk_size = st.sidebar.slider("Stream Chunk Size (Rows)", 10000, 200000, 50000, step=10000)

if not Path(csv_file_path).is_file():
    st.info(f"👉 Please ensure your dataset is placed at `{csv_file_path}` to start processing.")
else:
    # Read shallow data headers to extract structural mapping targets safely
    header_df = pd.read_csv(csv_file_path, nrows=100)
    all_columns = list(header_df.columns)

    st.sidebar.header("🛡️ Fairness Mapping Parameters")
    protected_attr = st.sidebar.selectbox("Protected Attribute (A)", all_columns, index=min(1, len(all_columns)-1))
    
    # Extract unique values from sample safely to establish reference baselines
    unique_vals = list(header_df[protected_attr].dropna().unique())
    privileged_val = st.sidebar.selectbox("Privileged Group Baseline (a')", unique_vals if unique_vals else ["Select"])
    
    target_col = st.sidebar.selectbox("Target Label (Y)", all_columns, index=min(2, len(all_columns)-1))
    prediction_col = st.sidebar.selectbox("Model Prediction Feature (Ŷ)", all_columns, index=min(2, len(all_columns)-1))

    # Build Configuration Context
    config = StreamingDataConfig(
        filepath=csv_file_path,
        protected_attribute=protected_attr,
        privileged_value=privileged_val,
        target_column=target_col,
        chunk_size=chunk_size
    )

    # Core Navigation Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Memory-Safe Profiling", "✨ Streaming Preprocessing", "⚖️ Fairness Disparity Audits"])

    with tab1:
        st.subheader("🔢 Incremental Mathematical Data Profile")
        st.markdown("Generates robust global statistics and tracks dataset completeness across all rows iteratively.")
        
        if st.button("Execute Global Profile Analytics"):
            with st.spinner("Streaming data segments out-of-core..."):
                profiler = OutOfCoreProfiler(config)
                stats_df, total_rows = profiler.compute_global_numerical_metrics()
                
                st.success(f"Successfully processed {total_rows:,} total rows across data chunks!")
                st.dataframe(stats_df, use_container_width=True)

    with tab2:
        st.subheader("🛠️ Memory-Safe Data Preprocessing")
        st.markdown("Imputes missing records using global summaries, performs standard scaling, and encodes fairness identifiers.")
        out_csv = st.text_input("Output CSV Filename", value="processed_output_data.csv")
        
        if st.button("Run Transformation Pipeline"):
            with st.spinner("Processing stream pipeline sequentially..."):
                profiler = OutOfCoreProfiler(config)
                stats_df, _ = profiler.compute_global_numerical_metrics()
                
                transformer = LowMemoryPreprocessor(config, stats_df)
                transformer.process_and_save_stream(out_csv)
                st.success(f"🎉 Preprocessed data stream saved successfully to `{out_csv}`!")

    with tab3:
        st.subheader("🕵️ Algorithmic Bias & Socio-Technical Audit")
        st.markdown("Computes exact disparity indicators incrementally across the dataset to uncover bias vectors [1].")
        
        if st.button("Run Fairness Risk Audit"):
            with st.spinner("Auditing stream parity vectors..."):
                auditor = StreamingFairnessAuditor(config, prediction_col)
                fairness_metrics = auditor.execute_audit()

                # Metric Cards Dashboard
                c1, c2, c3 = st.columns(3)
                c1.metric("Demographic Parity Diff", f"{fairness_metrics['Demographic Parity Diff']:.4f}")
                c2.metric("True Positive Rate Diff (EO)", f"{fairness_metrics['True Positive Rate Diff']:.4f}")
                c3.metric("False Positive Rate Diff (EO)", f"{fairness_metrics['False Positive Rate Diff']:.4f}")

                st.markdown(f"**Disparate Impact Ratio:** `{fairness_metrics['Disparate Impact Ratio']:.4f}`")
                
                # Verify compliance against the 80% rule
                if 0.8 <= fairness_metrics['Disparate Impact Ratio'] <= 1.25:
                    st.success("✅ Pass: Selection ratio adheres to the standard 80% legal fairness rule.")
                else:
                    st.error("⚠️ Adverse Impact Flagged: Selection ratio violates the 80% legal rule.")

                # Render Visual Evaluation Chart Container
                st.markdown("### 📊 Fairness Disparity Map")
                fig, ax = plt.subplots(figsize=(7, 3.5))
                plot_keys = ["Demographic Parity Diff", "True Positive Rate Diff", "False Positive Rate Diff"]
                plot_vals = [fairness_metrics[k] for k in plot_keys]
                
                colors = ["#E74C3C" if abs(v) > 0.1 else "#2ECC71" for v in plot_vals]
                sns.barplot(x=plot_vals, y=plot_keys, palette=colors, ax=ax, hue=plot_keys, legend=False)
                ax.axvline(0, color="black", linestyle="--", linewidth=1)
                ax.axvspan(-0.1, 0.1, color="green", alpha=0.08)
                ax.set_xlim(-1, 1)
                ax.set_xlabel("Disparity Score (<0 Favors Privileged | >0 Favors Protected)")
                ax.set_title("Fairness Metric Disparities (Ideal Parity at 0.0)")
                
                st.pyplot(fig)
