import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/strava.csv", parse_dates=["timestamp"])
    return df

df = load_data()

# Recreate derived columns
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["month_name"] = df["timestamp"].dt.strftime('%B')
df["time_of_day"] = pd.cut(
    df["timestamp"].dt.hour,
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"],
    right=False
)

# Sidebar filters
st.sidebar.header("Filters")
time_options = df["time_of_day"].dropna().unique().tolist()
month_options = df["month_name"].dropna().unique().tolist()

selected_time = st.sidebar.multiselect("Time of Day", options=time_options, default=time_options)
selected_months = st.sidebar.multiselect("Month", options=month_options, default=month_options)

metrics = [
    "Vertical Oscillation", "Cadence", "Power", "Air Power",
    "Ground Time", "Form Power", "Leg Spring Stiffness", "heart_rate", "speed"
]
selected_metric = st.sidebar.selectbox("Select Metric to Visualize", metrics)

remove_outliers = st.sidebar.checkbox("Remove outliers", value=True)

# Filter data
df_filtered = df[
    df["time_of_day"].isin(selected_time) &
    df["month_name"].isin(selected_months)
]

# Outlier handling
if remove_outliers and not df_filtered[selected_metric].dropna().empty:
    Q1 = df_filtered[selected_metric].quantile(0.25)
    Q3 = df_filtered[selected_metric].quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df_filtered[
        (df_filtered[selected_metric] >= Q1 - 1.5 * IQR) &
        (df_filtered[selected_metric] <= Q3 + 1.5 * IQR)
    ]

# App title
st.title("Strava Wearable Metrics Explorer")
st.markdown("Explore cadence, power, vertical oscillation, and more by time of day and month.")

# Empty filter fallback
if df_filtered.empty:
    st.info("No data matched your filters. Showing fallback histogram for full dataset.")
    fig_fallback, ax_fallback = plt.subplots()
    sns.histplot(df[selected_metric], bins=30, kde=True, ax=ax_fallback)
    st.pyplot(fig_fallback)
    st.stop()

# Layout Tabs
tab1, tab2, tab3 = st.tabs(["📈 KDE & Boxplot", "🔍 Scatterplot", "📊 Histogram Only"])

# === Tab 1: KDE + Boxplot ===
with tab1:
    st.subheader(f"Density Plot of {selected_metric}")
    if df_filtered[selected_metric].dropna().empty:
        st.warning("No valid data for KDE plot.")
    else:
        fig1, ax1 = plt.subplots()
        sns.kdeplot(
            data=df_filtered,
            x=selected_metric,
            hue="time_of_day",
            common_norm=False,
            fill=True,
            ax=ax1
        )
        st.pyplot(fig1)

    st.subheader(f"Boxplot of {selected_metric} by Time of Day")
    if df_filtered[selected_metric].dropna().empty or df_filtered["time_of_day"].dropna().nunique() < 2:
        st.warning("Not enough data for a boxplot.")
    else:
        fig2, ax2 = plt.subplots()
        sns.boxplot(
            data=df_filtered,
            x="time_of_day",
            y=selected_metric,
            palette="Set2",
            ax=ax2
        )
        st.pyplot(fig2)

# === Tab 2: Scatterplot (Custom X/Y + Regression) ===
with tab2:
    st.subheader("Custom Scatterplot")

    x_metric = st.selectbox("X-axis Metric", metrics, index=metrics.index("speed"))
    y_metric = st.selectbox("Y-axis Metric", metrics, index=metrics.index("heart_rate"))

    scatter_time = st.multiselect(
        "Time of Day", options=time_options, default=time_options, key="scatter_time"
    )
    scatter_month = st.multiselect(
        "Month", options=month_options, default=month_options, key="scatter_month"
    )
    show_reg = st.checkbox("Add regression line", value=False, key="scatter_reg")

    df_scatter = df[
        df["time_of_day"].isin(scatter_time) &
        df["month_name"].isin(scatter_month)
    ]

    st.markdown(f"**Plotting:** `{x_metric}` vs `{y_metric}`")

    if df_scatter[x_metric].dropna().empty or df_scatter[y_metric].dropna().empty:
        st.warning("Insufficient data for selected x/y metrics.")
    else:
        fig3, ax3 = plt.subplots()
        sns.scatterplot(
            data=df_scatter,
            x=x_metric,
            y=y_metric,
            hue="time_of_day",
            palette="husl",
            ax=ax3
        )
        if show_reg:
            sns.regplot(
                data=df_scatter,
                x=x_metric,
                y=y_metric,
                scatter=False,
                ax=ax3,
                color="gray",
                line_kws={"linestyle": "dashed"}
            )
        ax3.set_title(f"{y_metric} vs {x_metric}")
        st.pyplot(fig3)

# === Tab 3: Histogram ===
with tab3:
    st.subheader(f"Histogram of {selected_metric}")
    fig4, ax4 = plt.subplots()
    sns.histplot(df_filtered[selected_metric], bins=30, kde=True, ax=ax4)
    st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("Built by Alexis Parker · Powered by Streamlit")
