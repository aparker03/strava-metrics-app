from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Strava Metrics Explorer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = "data/strava.csv"
EXPECTED_METRICS = [
    "Vertical Oscillation",
    "Cadence",
    "Power",
    "Air Power",
    "Ground Time",
    "Form Power",
    "Leg Spring Stiffness",
    "heart_rate",
    "speed",
    "enhanced_speed",
    "distance",
    "altitude",
    "enhanced_altitude",
]
TIME_OF_DAY_ORDER = ["Night", "Morning", "Afternoon", "Evening"]
SEMICIRCLE_SCALE = 180 / 2**31
MAX_CHART_ROWS = 5_000
MAX_MAP_ROWS = 10_000
MAX_PREVIEW_ROWS = 1_000


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    """Load the bundled sample Strava export."""
    return pd.read_csv(DEFAULT_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_contents: bytes) -> pd.DataFrame:
    """Load a user-uploaded Strava CSV."""
    return pd.read_csv(BytesIO(file_contents))


def coerce_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = coerce_timestamp(df)
    if "timestamp" not in df.columns:
        return df

    df["date"] = df["timestamp"].dt.date
    df["month_name"] = df["timestamp"].dt.month_name()
    df["month_number"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["time_of_day"] = pd.cut(
        df["timestamp"].dt.hour,
        bins=[0, 6, 12, 18, 24],
        labels=TIME_OF_DAY_ORDER,
        right=False,
        include_lowest=True,
    )
    return df


def add_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"position_lat", "position_long"}.issubset(df.columns):
        lat = pd.to_numeric(df["position_lat"], errors="coerce")
        lon = pd.to_numeric(df["position_long"], errors="coerce")

        if lat.abs().max(skipna=True) > 90 or lon.abs().max(skipna=True) > 180:
            lat = lat * SEMICIRCLE_SCALE
            lon = lon * SEMICIRCLE_SCALE

        df["lat"] = lat
        df["lon"] = lon
    return df


def available_metrics(df: pd.DataFrame) -> list[str]:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    excluded = {"position_lat", "position_long", "lat", "lon", "month_number", "year"}
    ordered_expected = [metric for metric in EXPECTED_METRICS if metric in numeric_columns]
    remaining = [
        column
        for column in numeric_columns
        if column not in ordered_expected and column not in excluded and not column.startswith("unknown_")
    ]
    return ordered_expected + remaining


def sorted_months(df: pd.DataFrame) -> list[str]:
    if {"month_name", "month_number"}.issubset(df.columns):
        month_lookup = (
            df[["month_name", "month_number"]]
            .dropna()
            .drop_duplicates()
            .sort_values("month_number")
        )
        return month_lookup["month_name"].tolist()
    return []


def filter_data(
    df: pd.DataFrame,
    selected_time: list[str],
    selected_months: list[str],
    selected_years: list[int],
    date_range: tuple,
    selected_activities: list[str],
) -> pd.DataFrame:
    filtered = df.copy()

    if "time_of_day" in filtered.columns and selected_time:
        filtered = filtered[filtered["time_of_day"].astype(str).isin(selected_time)]

    if "month_name" in filtered.columns and selected_months:
        filtered = filtered[filtered["month_name"].isin(selected_months)]

    if "year" in filtered.columns and selected_years:
        filtered = filtered[filtered["year"].isin(selected_years)]

    if "timestamp" in filtered.columns and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start_date)
            & (filtered["timestamp"].dt.date <= end_date)
        ]

    if "datafile" in filtered.columns and selected_activities:
        filtered = filtered[filtered["datafile"].isin(selected_activities)]

    return filtered


def remove_metric_outliers(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns or df[metric].dropna().empty:
        return df

    q1 = df[metric].quantile(0.25)
    q3 = df[metric].quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return df

    return df[(df[metric] >= q1 - 1.5 * iqr) & (df[metric] <= q3 + 1.5 * iqr)]


def format_number(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"


def metric_mean(df: pd.DataFrame, metric: str) -> float:
    if metric not in df.columns:
        return float("nan")
    return pd.to_numeric(df[metric], errors="coerce").mean()


def make_activity_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if "datafile" not in df.columns or df.empty:
        return pd.DataFrame()

    aggregations = {metric: "mean" for metric in metrics if metric in df.columns}
    if "distance" in df.columns:
        aggregations["distance"] = "max"
    if "timestamp" in df.columns:
        aggregations["timestamp"] = ["min", "max", "count"]

    summary = df.groupby("datafile").agg(aggregations)
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.reset_index()

    rename_map = {
        "timestamp_min": "start_time",
        "timestamp_max": "end_time",
        "timestamp_count": "records",
        "distance_max": "distance",
    }
    summary = summary.rename(columns=rename_map)

    if {"start_time", "end_time"}.issubset(summary.columns):
        summary["duration_min"] = (
            summary["end_time"] - summary["start_time"]
        ).dt.total_seconds() / 60

    return summary.sort_values("start_time" if "start_time" in summary.columns else "datafile")


def dataframe_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def plot_empty_state(message: str) -> None:
    st.info(message)


def normalize_date_range(date_value, min_date, max_date) -> tuple:
    if isinstance(date_value, (tuple, list)):
        if len(date_value) == 2:
            return date_value[0], date_value[1]
        if len(date_value) == 1:
            return date_value[0], date_value[0]
    return min_date, max_date


def chart_sample(df: pd.DataFrame, limit: int = MAX_CHART_ROWS) -> pd.DataFrame:
    if len(df) <= limit:
        return df
    return df.sample(limit, random_state=42).sort_index()


def use_stretch_kwargs() -> dict:
    version_parts = []
    for part in st.__version__.split(".")[:2]:
        digits = "".join(character for character in part if character.isdigit())
        version_parts.append(int(digits or 0))

    if tuple(version_parts) >= (1, 50):
        return {"width": "stretch"}
    return {"use_container_width": True}


STRETCH = use_stretch_kwargs()


st.title("🏃 Strava Metrics Explorer")
st.markdown(
    "Turn Strava activity exports into an interactive performance dashboard with "
    "filters, trends, relationship analysis, route mapping, and downloadable data."
)

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload a Strava CSV", type=["csv"])
    st.caption("Upload a compatible export or use the bundled sample dataset.")

if uploaded_file is not None:
    df = load_uploaded_data(uploaded_file.getvalue())
    data_source = uploaded_file.name
else:
    df = load_default_data()
    data_source = DEFAULT_DATA_PATH

df = add_map_columns(add_derived_columns(df))

missing_required = [column for column in ["timestamp"] if column not in df.columns]
if missing_required:
    st.error(f"Missing required column(s): {', '.join(missing_required)}")
    st.stop()

metrics = available_metrics(df)
if not metrics:
    st.error("No numeric metrics were found in this dataset.")
    st.stop()

valid_timestamps = df["timestamp"].dropna()
if valid_timestamps.empty:
    st.error("The timestamp column could not be parsed. Please upload a CSV with valid timestamps.")
    st.stop()

min_date = valid_timestamps.min().date()
max_date = valid_timestamps.max().date()
time_options = [option for option in TIME_OF_DAY_ORDER if option in df["time_of_day"].astype(str).unique()]
month_options = sorted_months(df)
year_options = sorted(df["year"].dropna().astype(int).unique().tolist())
activity_options = sorted(df["datafile"].dropna().unique().tolist()) if "datafile" in df.columns else []

with st.sidebar:
    st.header("Filters")
    selected_metric = st.selectbox("Primary metric", metrics, index=0)
    comparison_metric = st.selectbox(
        "Comparison metric",
        metrics,
        index=metrics.index("heart_rate") if "heart_rate" in metrics else min(1, len(metrics) - 1),
    )
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    selected_years = st.multiselect("Year", options=year_options, default=year_options)
    selected_months = st.multiselect("Month", options=month_options, default=month_options)
    selected_time = st.multiselect("Time of day", options=time_options, default=time_options)
    selected_activities = st.multiselect(
        "Activities",
        options=activity_options,
        default=[],
        help="Leave empty to include every activity.",
    )
    remove_outliers = st.checkbox("Remove outliers for primary metric", value=True)

date_range = normalize_date_range(date_range, min_date, max_date)

filtered = filter_data(
    df=df,
    selected_time=selected_time,
    selected_months=selected_months,
    selected_years=selected_years,
    date_range=date_range,
    selected_activities=selected_activities,
)

if remove_outliers:
    filtered = remove_metric_outliers(filtered, selected_metric)

st.caption(f"Data source: `{data_source}`")

if filtered.empty:
    st.warning("No data matched the current filters. Try expanding the date, month, time, or activity selections.")
    st.stop()

activity_count = filtered["datafile"].nunique() if "datafile" in filtered.columns else None
record_count = len(filtered)
distance_value = metric_mean(filtered.groupby("datafile")["distance"].max().reset_index(), "distance") if {"datafile", "distance"}.issubset(filtered.columns) else metric_mean(filtered, "distance")

summary_cols = st.columns(5)
summary_cols[0].metric("Records", f"{record_count:,}")
summary_cols[1].metric("Activities", f"{activity_count:,}" if activity_count is not None else "—")
summary_cols[2].metric("Avg Heart Rate", format_number(metric_mean(filtered, "heart_rate"), 1))
summary_cols[3].metric("Avg Speed", format_number(metric_mean(filtered, "speed"), 2))
summary_cols[4].metric("Avg Power", format_number(metric_mean(filtered, "Power"), 1))

st.divider()

page = st.sidebar.radio(
    "Dashboard section",
    ["Overview", "Trends", "Distributions", "Relationships", "Activities", "Map", "Data"],
    help="Only the selected section is rendered, which keeps the app responsive on Streamlit Cloud.",
)

if page == "Overview":
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Performance snapshot")
        snapshot = pd.DataFrame(
            {
                "Metric": [
                    "Date range",
                    "Primary metric average",
                    "Primary metric median",
                    "Primary metric max",
                    "Comparison metric average",
                    "Average distance by activity",
                ],
                "Value": [
                    f"{filtered['timestamp'].min().date()} to {filtered['timestamp'].max().date()}",
                    format_number(metric_mean(filtered, selected_metric), 2),
                    format_number(filtered[selected_metric].median(), 2),
                    format_number(filtered[selected_metric].max(), 2),
                    format_number(metric_mean(filtered, comparison_metric), 2),
                    format_number(distance_value, 2),
                ],
            }
        )
        st.dataframe(snapshot, hide_index=True, **STRETCH)

    with right:
        st.subheader("Records by time of day")
        if "time_of_day" in filtered.columns:
            time_counts = filtered["time_of_day"].value_counts().reindex(TIME_OF_DAY_ORDER).dropna()
            st.bar_chart(time_counts, **STRETCH)
        else:
            plot_empty_state("No time-of-day column is available.")

    st.subheader("Monthly average")
    if {"month_number", "month_name", selected_metric}.issubset(filtered.columns):
        monthly = (
            filtered.groupby(["month_number", "month_name"], observed=True)[selected_metric]
            .mean()
            .reset_index()
            .sort_values("month_number")
            .set_index("month_name")
        )
        st.bar_chart(monthly[[selected_metric]], **STRETCH)
    else:
        plot_empty_state("Monthly averages are unavailable for this dataset.")

if page == "Trends":
    st.subheader(f"{selected_metric} over time")
    trend_frequency = st.radio(
        "Aggregation",
        options=["Raw records", "Daily average", "Weekly average"],
        horizontal=True,
    )

    trend_source = chart_sample(filtered, MAX_CHART_ROWS)
    if len(filtered) > len(trend_source):
        st.caption(f"Showing a deterministic sample of {len(trend_source):,} records for performance.")
    trend_data = trend_source[["timestamp", selected_metric]].dropna().sort_values("timestamp")
    if trend_data.empty:
        plot_empty_state("There is no valid data for the selected metric.")
    else:
        if trend_frequency == "Daily average":
            trend_data = trend_data.set_index("timestamp").resample("D").mean().dropna()
        elif trend_frequency == "Weekly average":
            trend_data = trend_data.set_index("timestamp").resample("W").mean().dropna()
        else:
            trend_data = trend_data.set_index("timestamp")
        st.line_chart(trend_data[[selected_metric]], **STRETCH)

if page == "Distributions":
    kde_col, box_col = st.columns(2)

    with kde_col:
        st.subheader(f"Density of {selected_metric}")
        if filtered[selected_metric].dropna().nunique() < 2:
            plot_empty_state("Not enough variation for a density plot.")
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            plot_data = chart_sample(filtered, MAX_CHART_ROWS)
            if len(filtered) > len(plot_data):
                st.caption(f"Plots use a deterministic sample of {len(plot_data):,} records for performance.")
            sns.kdeplot(
                data=plot_data,
                x=selected_metric,
                hue="time_of_day" if "time_of_day" in filtered.columns else None,
                common_norm=False,
                fill=True,
                ax=ax,
            )
            ax.set_xlabel(selected_metric)
            st.pyplot(fig, **STRETCH)

    with box_col:
        st.subheader(f"{selected_metric} by time of day")
        if "time_of_day" not in filtered.columns or filtered["time_of_day"].dropna().nunique() < 2:
            plot_empty_state("Not enough time-of-day groups for a boxplot.")
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            box_data = chart_sample(filtered, MAX_CHART_ROWS)
            sns.boxplot(
                data=box_data,
                x="time_of_day",
                y=selected_metric,
                hue="time_of_day",
                palette="Set2",
                legend=False,
                ax=ax,
            )
            ax.set_xlabel("Time of day")
            ax.set_ylabel(selected_metric)
            st.pyplot(fig, **STRETCH)

    st.subheader(f"Histogram of {selected_metric}")
    fig, ax = plt.subplots(figsize=(12, 5))
    hist_data = chart_sample(filtered, MAX_CHART_ROWS)
    sns.histplot(hist_data[selected_metric].dropna(), bins=30, kde=True, ax=ax)
    ax.set_xlabel(selected_metric)
    st.pyplot(fig, **STRETCH)

if page == "Relationships":
    scatter_col, corr_col = st.columns([1.1, 1])

    with scatter_col:
        st.subheader("Metric relationship")
        show_regression = st.checkbox("Add regression line", value=True)
        scatter_source = chart_sample(filtered, MAX_CHART_ROWS)
        if len(filtered) > len(scatter_source):
            st.caption(f"Scatterplot uses a deterministic sample of {len(scatter_source):,} records for performance.")
        scatter_data = scatter_source[[selected_metric, comparison_metric, "time_of_day"]].dropna()
        if scatter_data.empty:
            plot_empty_state("Insufficient data for the selected metrics.")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                data=scatter_data,
                x=selected_metric,
                y=comparison_metric,
                hue="time_of_day" if "time_of_day" in scatter_data.columns else None,
                palette="husl",
                alpha=0.7,
                ax=ax,
            )
            if show_regression and scatter_data[selected_metric].nunique() > 1:
                sns.regplot(
                    data=scatter_data,
                    x=selected_metric,
                    y=comparison_metric,
                    scatter=False,
                    ax=ax,
                    color="gray",
                    line_kws={"linestyle": "dashed"},
                )
            correlation = scatter_data[[selected_metric, comparison_metric]].corr().iloc[0, 1]
            ax.set_title(f"Correlation: {format_number(correlation, 2)}")
            st.pyplot(fig, **STRETCH)

    with corr_col:
        st.subheader("Correlation heatmap")
        corr_metrics = [metric for metric in metrics if metric in filtered.columns and filtered[metric].dropna().nunique() > 1]
        corr_metrics = corr_metrics[:10]
        if len(corr_metrics) < 2:
            plot_empty_state("At least two populated numeric metrics are needed for a heatmap.")
        else:
            corr = chart_sample(filtered, MAX_CHART_ROWS)[corr_metrics].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
            st.pyplot(fig, **STRETCH)

if page == "Activities":
    st.subheader("Activity-level summary")
    activity_summary = make_activity_summary(filtered, metrics)
    if activity_summary.empty:
        plot_empty_state("No activity identifier was found in this dataset.")
    else:
        st.dataframe(activity_summary, **STRETCH, hide_index=True)
        st.download_button(
            "Download activity summary",
            data=dataframe_csv(activity_summary),
            file_name="strava_activity_summary.csv",
            mime="text/csv",
        )

if page == "Map":
    st.subheader("Route map")
    if {"lat", "lon"}.issubset(filtered.columns):
        map_data = filtered[["lat", "lon"]].dropna()
        map_data = map_data[(map_data["lat"].between(-90, 90)) & (map_data["lon"].between(-180, 180))]
        if map_data.empty:
            plot_empty_state("No valid latitude/longitude points are available for the current filters.")
        else:
            if len(map_data) > MAX_MAP_ROWS:
                map_data = map_data.sample(MAX_MAP_ROWS, random_state=42).sort_index()
                st.caption(f"Map uses a deterministic sample of {MAX_MAP_ROWS:,} points for performance.")
            st.map(map_data, **STRETCH)
            st.caption("FIT semicircle coordinates are converted to latitude/longitude when needed.")
    else:
        plot_empty_state("This dataset does not include position_lat and position_long columns.")

if page == "Data":
    st.subheader("Filtered data")
    preview = filtered.head(MAX_PREVIEW_ROWS)
    if len(filtered) > len(preview):
        st.caption(f"Previewing the first {len(preview):,} of {len(filtered):,} filtered records. Download includes all rows.")
    st.dataframe(preview, **STRETCH, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        data=dataframe_csv(filtered),
        file_name="strava_filtered_data.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown("Built by Alexis Parker · Powered by Streamlit")
