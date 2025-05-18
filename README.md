# Strava Metrics Explorer
[![View on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movement-mapped.streamlit.app/)

Visualize wearable data from Strava using Streamlit. Includes filters, KDE plots, boxplots, and scatterplots with regression.

## Features

- Filter by time of day and month  
- Toggle outlier removal  
- KDE density plots and boxplots  
- Custom scatterplots with optional regression line  
- Univariate histograms for fallback views  
- Built with Streamlit, Seaborn, Matplotlib, and Pandas

## Demo

[Live App on Streamlit Cloud](https://YOUR-USERNAME.streamlit.app)

## Getting Started

To run the app locally:

1. Clone this repository  
2. Install dependencies  
3. Launch the app

```bash
pip install -r requirements.txt
streamlit run strava_app.py
```
## Data

This project uses exported Strava activity data from a personal device.  
It includes metrics such as cadence, power, heart rate, and vertical oscillation.

## Author

**Alexis Parker**  
Master of Applied Data Science, University of Michigan
