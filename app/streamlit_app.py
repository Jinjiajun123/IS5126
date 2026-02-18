"""
streamlit_app.py — Hotel Analytics Dashboard

Run:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import DB_PATH, SAMPLE_DB_PATH, RATING_COLUMNS, RATING_LABELS, get_db_connection

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Analytics Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; font-size: 0.9rem; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ── DB helper ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_data(db_path: str):
    """Load all reviews into a DataFrame (cached)."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM reviews", conn)
    df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
    conn.close()
    return df


@st.cache_data(ttl=600)
def load_hotels(db_path: str):
    import sqlite3
    conn = sqlite3.connect(db_path)
    hotels = pd.read_sql_query("SELECT * FROM hotels ORDER BY num_reviews DESC", conn)
    conn.close()
    return hotels


def detect_db():
    """Use full DB if available, else sample."""
    if DB_PATH.exists():
        return str(DB_PATH)
    return str(SAMPLE_DB_PATH)


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hotel-building.png", width=64)
st.sidebar.title("🏨 Hotel Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Hotel Explorer", "📈 Trend Analysis",
     "🏆 Competitive Benchmarking", "⭐ Satisfaction Drivers"],
)

db_path = detect_db()
df = load_data(db_path)
hotels = load_hotels(db_path)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Dashboard Overview")
    st.caption("Key metrics at a glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{len(df):,}")
    c2.metric("Total Hotels", f"{df['hotel_id'].nunique():,}")
    c3.metric("Avg Overall Rating", f"{df['rating_overall'].mean():.2f}")
    c4.metric("Date Range",
              f"{df['date_parsed'].min().strftime('%Y-%m')} – {df['date_parsed'].max().strftime('%Y-%m')}")

    st.divider()

    # Rating distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rating Distribution")
        fig = px.histogram(df, x="rating_overall", nbins=5, color_discrete_sequence=["#667eea"],
                           labels={"rating_overall": "Overall Rating"})
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Reviews per Year")
        year_counts = df.groupby("year").size().reset_index(name="count")
        fig = px.bar(year_counts, x="year", y="count", color_discrete_sequence=["#764ba2"],
                     labels={"year": "Year", "count": "Reviews"})
        st.plotly_chart(fig, use_container_width=True)

    # Top hotels
    st.subheader("Top 10 Hotels by Average Rating (min 30 reviews)")
    top = (df.groupby("hotel_id")
           .agg(avg_rating=("rating_overall", "mean"), reviews=("rating_overall", "count"))
           .query("reviews >= 30")
           .nlargest(10, "avg_rating")
           .reset_index())
    top["avg_rating"] = top["avg_rating"].round(2)
    st.dataframe(top, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: HOTEL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Hotel Explorer":
    st.title("🔍 Hotel Explorer")

    hotel_ids = sorted(df["hotel_id"].unique())
    selected = st.selectbox("Select Hotel ID", hotel_ids)

    hdf = df[df["hotel_id"] == selected]
    st.metric("Reviews for this hotel", f"{len(hdf):,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rating Profile")
        means = hdf[RATING_COLUMNS].mean()
        labels = [RATING_LABELS[c] for c in RATING_COLUMNS]
        fig = go.Figure(go.Scatterpolar(
            r=means.values.tolist() + [means.values[0]],
            theta=labels + [labels[0]],
            fill="toself", fillcolor="rgba(102,126,234,0.25)",
            line=dict(color="#667eea", width=2),
        ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[1, 5])), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Rating Trend Over Time")
        monthly = hdf.set_index("date_parsed").resample("M")["rating_overall"].mean().reset_index()
        fig = px.line(monthly, x="date_parsed", y="rating_overall",
                      labels={"date_parsed": "Date", "rating_overall": "Avg Overall Rating"},
                      color_discrete_sequence=["#764ba2"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Reviews")
    recent = hdf.nlargest(10, "date_parsed")[["date", "title", "rating_overall", "text"]]
    st.dataframe(recent, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Analysis":
    st.title("📈 Trend Analysis")

    # Year filter
    years = sorted(df["year"].dropna().unique())
    selected_years = st.slider("Year Range", int(min(years)), int(max(years)),
                               (int(min(years)), int(max(years))))
    mask = df["year"].between(*selected_years)
    filtered = df[mask]

    st.subheader("Monthly Average Ratings Over Time")
    rating_choice = st.selectbox("Rating Dimension", list(RATING_LABELS.values()))
    col_name = [k for k, v in RATING_LABELS.items() if v == rating_choice][0]

    monthly = (filtered.set_index("date_parsed")
               .resample("M")[col_name].agg(["mean", "count"]).reset_index())
    fig = px.line(monthly, x="date_parsed", y="mean",
                  labels={"date_parsed": "Date", "mean": f"Avg {rating_choice}"},
                  color_discrete_sequence=["#667eea"])
    fig.add_bar(x=monthly["date_parsed"], y=monthly["count"], name="Review Count",
                opacity=0.2, marker_color="#FF9800", yaxis="y2")
    fig.update_layout(
        yaxis2=dict(title="Review Count", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Year-over-Year Comparison")
    yearly = filtered.groupby("year")[RATING_COLUMNS].mean().reset_index()
    yearly_melted = yearly.melt(id_vars="year", var_name="dimension", value_name="avg_rating")
    yearly_melted["dimension"] = yearly_melted["dimension"].map(RATING_LABELS)
    fig = px.bar(yearly_melted, x="dimension", y="avg_rating", color="year",
                 barmode="group", color_continuous_scale="Viridis",
                 labels={"dimension": "", "avg_rating": "Avg Rating"})
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: COMPETITIVE BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Competitive Benchmarking":
    st.title("🏆 Competitive Benchmarking")
    st.caption("Hotels clustered into peer groups via K-Means on rating profiles")

    from src.benchmarking import (
        compute_hotel_features, cluster_hotels,
        analyze_group_performance, generate_recommendations,
    )

    n_clusters = st.sidebar.slider("Number of Clusters", 3, 10, 5)
    min_rev = st.sidebar.slider("Min Reviews per Hotel", 5, 100, 10)

    with st.spinner("Computing clusters…"):
        features = compute_hotel_features(db_path=Path(db_path), min_reviews=min_rev)
        df_cl, sil, _, _ = cluster_hotels(features, n_clusters=n_clusters)

    st.metric("Silhouette Score", f"{sil:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cluster Sizes")
        sizes = df_cl["cluster"].value_counts().sort_index().reset_index()
        sizes.columns = ["Cluster", "Hotels"]
        fig = px.bar(sizes, x="Cluster", y="Hotels", color="Cluster",
                     color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cluster Profiles")
        rating_cols = [c for c in df_cl.columns if c.startswith("avg_rating_")]
        cluster_means = df_cl.groupby("cluster")[rating_cols].mean()
        labels_short = [c.replace("avg_rating_", "").replace("_", " ").title()
                        for c in rating_cols]

        fig = go.Figure()
        for cid, row in cluster_means.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row.tolist() + [row.iloc[0]],
                theta=labels_short + [labels_short[0]],
                fill="toself", name=f"Cluster {cid}",
            ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[2, 5])), height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Group Performance Summary")
    summary = analyze_group_performance(df_cl)
    st.dataframe(summary.round(2), use_container_width=True)

    st.subheader("Recommendations for Under-Performers")
    recs = generate_recommendations(df_cl)
    if len(recs) > 0:
        st.dataframe(recs.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("All hotels are performing at or above their peer group average.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: SATISFACTION DRIVERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⭐ Satisfaction Drivers":
    st.title("⭐ Satisfaction Drivers")
    st.caption("Which rating dimensions drive overall satisfaction?")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation with Overall Rating")
        corr = df[RATING_COLUMNS].corr()["rating_overall"].drop("rating_overall").sort_values(ascending=False)
        corr.index = [RATING_LABELS.get(c, c) for c in corr.index]
        fig = px.bar(x=corr.values, y=corr.index, orientation="h",
                     labels={"x": "Correlation", "y": ""},
                     color=corr.values, color_continuous_scale="YlOrRd")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Correlation Heatmap")
        corr_full = df[RATING_COLUMNS].corr()
        labels = [RATING_LABELS[c] for c in RATING_COLUMNS]
        fig = px.imshow(corr_full.values, x=labels, y=labels,
                        color_continuous_scale="YlOrRd", text_auto=".2f",
                        aspect="auto")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Ratings by Dimension")
    avg_ratings = df[RATING_COLUMNS].mean().sort_values()
    avg_ratings.index = [RATING_LABELS[c] for c in avg_ratings.index]
    fig = px.bar(x=avg_ratings.values, y=avg_ratings.index, orientation="h",
                 labels={"x": "Avg Rating", "y": ""},
                 color=avg_ratings.values, color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mobile vs Desktop Satisfaction")
    mobile_comp = df.groupby("via_mobile")[RATING_COLUMNS].mean().T
    mobile_comp.columns = ["Desktop", "Mobile"]
    mobile_comp.index = [RATING_LABELS[c] for c in RATING_COLUMNS]
    st.dataframe(mobile_comp.round(3), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Hotel Analytics Dashboard · Assignment 1")
