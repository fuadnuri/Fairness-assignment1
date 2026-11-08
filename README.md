# Fairness Assignment 1

A Streamlit application for out-of-core exploratory data analysis, preprocessing, and fairness auditing of the Global Health Expenditure Database (GHED) CSV data.

## Requirements

- Python 3.10 or later
- The GHED dataset as a CSV file at `data/GHED_data.csv` (the default input path)

## Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the application

From the project directory, start Streamlit:

```bash
streamlit run app.py
```

The app expects the input CSV at `data/GHED_data.csv` by default. Change **CSV Target Path** in the sidebar to use another file. Adjust the stream chunk size there to control the number of rows processed at a time.

## Features

- **Memory-safe profiling:** Computes numerical summaries and missing-value counts by streaming CSV chunks.
- **Preprocessing:** Fills numeric missing values with global means, fills categorical missing values with `UNKNOWN_VAL`, standardizes numeric columns, and adds `is_privileged` and `target_label` mappings. The output path defaults to `processed_output_data.csv` in the project directory.
- **Fairness audit:** Compares the selected protected and remaining groups using demographic parity difference, disparate impact ratio, and true/false positive rate differences.

Choose the protected attribute, privileged value, target label, and prediction feature in the sidebar. The application reads column names from the selected input CSV, so these options depend on the dataset. Auditing and preprocessing scan the input in chunks; preprocessing writes its output CSV locally.
