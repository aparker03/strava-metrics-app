# Strava Metrics Explorer

[![View on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movement-mapped.streamlit.app/)

A polished Streamlit dashboard for exploring Strava and wearable activity data. The app turns CSV exports into interactive performance insights with summary cards, filters, trends, distributions, relationships, route mapping, activity summaries, and downloadable datasets.

## Highlights

- Upload your own Strava-compatible CSV or use the bundled sample data.
- Filter by date range, year, month, time of day, and activity file.
- Review dashboard summary cards for records, activities, heart rate, speed, and power.
- Explore trend charts with raw, daily, or weekly aggregation.
- Compare metric distributions with KDE plots, boxplots, and histograms.
- Analyze relationships with scatterplots, optional regression lines, and a correlation heatmap.
- Summarize each activity in a sortable table and download the results.
- Map route coordinates when `position_lat` and `position_long` are available.
- Download the filtered dataset for further analysis.

## Demo

[Live App on Streamlit Cloud](https://movement-mapped.streamlit.app/)

## Project Structure

```text
.
├── strava_app.py        # Streamlit dashboard
├── requirements.txt     # Runtime dependencies
├── data/
│   └── strava.csv       # Bundled sample Strava export
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd strava-metrics-app
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run strava_app.py
```

## Data Format

The app expects a CSV with a `timestamp` column. It automatically detects numeric columns for visualization, but it works best when the export includes common Strava/FIT fields such as:

- `datafile`
- `timestamp`
- `distance`
- `heart_rate`
- `speed`
- `enhanced_speed`
- `Cadence`
- `Power`
- `Vertical Oscillation`
- `Ground Time`
- `position_lat`
- `position_long`

If latitude and longitude values are stored as FIT semicircles, the app converts them into standard map coordinates.

## Using Your Own Data

1. Start the app.
2. Use the sidebar file uploader to select a CSV.
3. Adjust filters and metric selectors.
4. Download filtered data or activity summaries from the dashboard.

If no file is uploaded, the app falls back to `data/strava.csv`.

## Deployment on Streamlit Cloud

1. Push this repository to GitHub.
2. Create a new app in Streamlit Cloud.
3. Select this repository and branch.
4. Use `strava_app.py` as the app entry point.
5. Confirm Streamlit installs dependencies from `requirements.txt`.

## Maintenance Notes

Dependencies use version ranges to keep installs current while avoiding unexpected major-version upgrades. To refresh dependencies locally:

```bash
python -m pip install --upgrade -r requirements.txt
```

If you want fully reproducible installs, generate a lock file from a known-good environment:

```bash
python -m pip freeze > requirements-lock.txt
```

## Author

**Alexis Parker**  
Master of Applied Data Science, University of Michigan
