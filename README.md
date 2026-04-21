# AI Growth Analyst

An AI-powered revenue and marketing insight engine that automatically
analyzes business data, detects anomalies, and generates plain-English
recommendations — no analyst required.

## What it does

- Tracks revenue, CAC, ROAS, conversion rate, AOV daily
- Detects anomalies using rolling statistical thresholds
- Generates AI narratives and action recommendations via Gemini
- Exports shareable daily insight reports

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Gemini API key (free at aistudio.google.com)
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Run the ETL pipeline (builds all 4 analytical tables)
python -m etl.pipeline

# 4. Launch the dashboard
streamlit run app/dashboard.py
```

## Project structure

```
etl/          Raw CSV ingestion, cleaning, transformation to Parquet
analytics/    Metric computation, anomaly detection, insight generation
ai/           Gemini API integration for narratives and recommendations
reports/      Daily text report generator
app/          Streamlit dashboard
data/raw/     Original Kaggle CSVs (not committed)
data/processed/ Parquet analytical tables (not committed)
```

## Deploying to Streamlit Cloud

1. Push repo to GitHub (data/raw and data/processed are gitignored)
2. Connect repo at share.streamlit.io
3. Set `GEMINI_API_KEY` in the Streamlit Cloud secrets manager
4. Set main file path to `app/dashboard.py`

Note: Add a `setup.sh` or `packages.txt` if you need the ETL to run
on deploy. For a demo, commit the processed Parquet files directly
by removing `data/processed/` from `.gitignore`.

## Tech stack

| Layer       | Tool                     |
|-------------|--------------------------|
| Language    | Python 3.11              |
| Processing  | pandas, NumPy            |
| Storage     | Parquet (pyarrow)        |
| AI          | Google Gemini 1.5 Flash  |
| Frontend    | Streamlit                |
| Hosting     | Streamlit Community Cloud|
| Charts      | Plotly                   |
