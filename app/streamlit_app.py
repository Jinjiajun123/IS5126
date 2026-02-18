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
    page_title="Hotel Performance Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Font & Background */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .main .block-container { padding-top: 2rem; }
    
    /* Headings */
    h1, h2, h3 {
        color: #2D3748;
        font-weight: 600;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E2E8F0;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1A202C;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# ── DB helper ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_data(db_path: str):
    """Load all reviews into a DataFrame (cached)."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    # Enforce 100k limit also on read if DB is huge, though build process handles it.
    df = pd.read_sql_query("SELECT * FROM reviews LIMIT 100000", conn)
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
st.sidebar.image("app/logo.png", width=120)
st.sidebar.title("Hotel Analytics")
page = st.sidebar.radio(
    "Menu",
    ["📊 Executive Overview", "🔍 Hotel Explorer", "🔎 Global Search", "📈 Trends",
     "🏆 Market Positioning", "⭐ Customer Priorities", "🧠 Sentiment Analysis"],
)

db_path = detect_db()
df = load_data(db_path)
hotels = load_hotels(db_path)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.title("Executive Overview")
    st.caption("High-level performance snapshot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews Analyzed", f"{len(df):,}")
    c2.metric("Hotels Monitored", f"{df['hotel_id'].nunique():,}")
    c3.metric("Average Rating", f"{df['rating_overall'].mean():.2f} / 5.0")
    c4.metric("Data Period",
              f"{df['date_parsed'].min().strftime('%b %Y')} – {df['date_parsed'].max().strftime('%b %Y')}")

    st.divider()

    # Simple Visuals
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Rating Distribution")
        fig = px.histogram(df, x="rating_overall", nbins=5, 
                           color_discrete_sequence=["#4FD1C5"],
                           title="Distribution of Overall Ratings")
        fig.update_layout(bargap=0.2, showlegend=False, xaxis_title="Rating (1-5)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Review Volume Trend")
        year_counts = df.groupby("year").size().reset_index(name="count")
        fig = px.bar(year_counts, x="year", y="count", 
                     color_discrete_sequence=["#667EEA"],
                     title="Reviews Collected per Year")
        fig.update_layout(xaxis_title="Year", yaxis_title="Review Count")
        st.plotly_chart(fig, use_container_width=True)

    # Top hotels
    st.subheader("Top Performing Hotels")
    st.markdown("_Hotels with the highest average rating (minimum 30 reviews)_")
    top = (df.groupby("hotel_id")
           .agg(avg_rating=("rating_overall", "mean"), reviews=("rating_overall", "count"))
           .query("reviews >= 30")
           .nlargest(10, "avg_rating")
           .reset_index())
    top["avg_rating"] = top["avg_rating"].round(2)
    top.columns = ["Hotel ID", "Average Rating", "Review Count"]
    st.dataframe(top, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: HOTEL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: HOTEL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Hotel Explorer":
    st.title("Hotel Explorer")
    st.markdown("_Deep dive into specific hotel performance_")

    hotel_ids = sorted(df["hotel_id"].unique())
    selected = st.selectbox("Select Hotel ID", hotel_ids)

    hdf = df[df["hotel_id"] == selected]
    st.info(f"Analyzing **{len(hdf):,} reviews** for Hotel **{selected}**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance Profile")
        means = hdf[RATING_COLUMNS].mean()
        labels = [RATING_LABELS[c] for c in RATING_COLUMNS]
        fig = go.Figure(go.Scatterpolar(
            r=means.values.tolist() + [means.values[0]],
            theta=labels + [labels[0]],
            fill="toself", fillcolor="rgba(102,126,234,0.3)",
            line=dict(color="#5A67D8", width=3),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[1, 5], visible=True)), 
            height=400,
            margin=dict(t=20, b=20, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Rating History")
        monthly = hdf.set_index("date_parsed").resample("M")["rating_overall"].mean().reset_index()
        fig = px.line(monthly, x="date_parsed", y="rating_overall",
                      labels={"date_parsed": "Date", "rating_overall": "Avg Rating"},
                      color_discrete_sequence=["#9F7AEA"])
        fig.update_layout(yaxis_range=[1, 5])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Customer Feedback")
    recent = hdf.nlargest(5, "date_parsed")[["date", "title", "rating_overall", "text"]]
    recent.columns = ["Date", "Title", "Rating", "Review"]
    for i, row in recent.iterrows():
        with st.container():
            st.markdown(f"**{'⭐' * int(row['Rating'])}** — *{row['Date']}*")
            st.markdown(f"**{row['Title']}**")
            st.markdown(f"> {row['Review']}")
            st.divider()


    for i, row in recent.iterrows():
        with st.container():
            st.markdown(f"**{'⭐' * int(row['Rating'])}** — *{row['Date']}*")
            st.markdown(f"**{row['Title']}**")
            st.markdown(f"> {row['Review']}")
            st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GLOBAL SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Global Search":
    st.title("Global Review Search")
    st.markdown("_Find specific feedback across all hotels_")

    query = st.text_input("Enter keywords (e.g., 'breakfast', 'front desk', 'cleanliness')")

    if query:
        # Search in text and title
        mask = df["text"].str.contains(query, case=False, na=False) | \
               df["title"].str.contains(query, case=False, na=False)
        results = df[mask]

        st.info(f"Found **{len(results):,}** reviews matching '**{query}**'")

        if not results.empty:
            st.caption("Showing top 100 most recent matches:")
            display_cols = ["date_parsed", "hotel_id", "rating_overall", "title", "text"]
            
            # Format for display
            display_df = results.sort_values("date_parsed", ascending=False).head(100)[display_cols].copy()
            display_df["date_parsed"] = display_df["date_parsed"].dt.strftime("%Y-%m-%d")
            display_df.columns = ["Date", "Hotel ID", "Rating", "Title", "Review Text"]
            
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)
    else:
        st.markdown("Use the search bar above to find reviews containing specific words or phrases.")
        st.markdown("Example searches:")
        st.markdown("- `bed bugs`")
        st.markdown("- `manager`")
        st.markdown("- `noisy`")
        st.markdown("- `excellent service`")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends":
    st.title("Market Trends")
    st.markdown("_Track performance metrics over time_")

    # Metrics selector
    st.subheader("Monthly Performance Trends")
    rating_choice = st.selectbox("Select Metric", list(RATING_LABELS.values()))
    col_name = [k for k, v in RATING_LABELS.items() if v == rating_choice][0]

    monthly = (df.set_index("date_parsed")
               .resample("M")[col_name].agg(["mean", "count"]).reset_index())
    
    fig = px.line(monthly, x="date_parsed", y="mean",
                  labels={"date_parsed": "Date", "mean": f"Avg {rating_choice}"},
                  color_discrete_sequence=["#4299E1"])
    
    # Dual axis for volume
    fig.add_bar(x=monthly["date_parsed"], y=monthly["count"], name="Review Volume",
                opacity=0.15, marker_color="#ED8936", yaxis="y2")
    
    fig.update_layout(
        title=f"Trend for {rating_choice}",
        yaxis2=dict(title="Review Volume", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Year-over-Year Performance")
    yearly = df.groupby("year")[RATING_COLUMNS].mean().reset_index()
    yearly_melted = yearly.melt(id_vars="year", var_name="dimension", value_name="avg_rating")
    yearly_melted["dimension"] = yearly_melted["dimension"].map(RATING_LABELS)
    
    fig = px.bar(yearly_melted, x="dimension", y="avg_rating", color="year",
                 barmode="group", color_continuous_scale="Purples",
                 labels={"dimension": "", "avg_rating": "Avg Rating"})
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: COMPETITIVE BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MARKET POSITIONING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Market Positioning":
    st.title("Market Positioning")
    st.markdown("_Benchmark hotels against similar market segments_")

    from src.benchmarking import (
        compute_hotel_features, cluster_hotels,
        analyze_group_performance, generate_recommendations,
    )

    # Simplified controls (hidden from main view or fixed)
    n_clusters = 5
    min_rev = 20

    with st.spinner("Analyzing market segments..."):
        features = compute_hotel_features(db_path=Path(db_path), min_reviews=min_rev)
        df_cl, sil, _, _ = cluster_hotels(features, n_clusters=n_clusters)

    st.success(f"Identified {n_clusters} distinct market segments based on guest feedback patterns.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Market Segment Distribution")
        sizes = df_cl["cluster"].value_counts().sort_index().reset_index()
        sizes.columns = ["Segment", "Hotels"]
        sizes["Segment"] = sizes["Segment"].apply(lambda x: f"Segment {x+1}")
        
        fig = px.bar(sizes, x="Segment", y="Hotels", color="Segment",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Segment Characteristics")
        st.info("How different segments perform across key metrics")
        
        rating_cols = [c for c in df_cl.columns if c.startswith("avg_rating_")]
        cluster_means = df_cl.groupby("cluster")[rating_cols].mean()
        labels_short = [c.replace("avg_rating_", "").replace("_", " ").title() for c in rating_cols]

        fig = go.Figure()
        for cid, row in cluster_means.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row.tolist() + [row.iloc[0]],
                theta=labels_short + [labels_short[0]],
                fill="toself", name=f"Segment {cid+1}",
            ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[2, 5], visible=True)), height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strategic Recommendations")
    st.markdown("Actionable insights for hotels underperforming in their segment:")
    recs = generate_recommendations(df_cl)
    if len(recs) > 0:
        # Clean up recommendation table for display
        display_recs = recs.copy()
        display_recs.columns = [c.replace("_", " ").title() for c in display_recs.columns]
        st.dataframe(display_recs.head(15), use_container_width=True, hide_index=True)
    else:
        st.info("All hotels are performing well within their market segments.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: CUSTOMER PRIORITIES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⭐ Customer Priorities":
    st.title("Customer Priorities")
    st.markdown("_What drives guest satisfaction?_")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Key Satisfaction Drivers")
        st.markdown(
            "This chart shows which factors correlate most strongly with overall satisfaction. "
            "**Focus on improving the top bars to boost overall ratings.**"
        )
        
        corr = df[RATING_COLUMNS].corr()["rating_overall"].drop("rating_overall").sort_values(ascending=True)
        corr.index = [RATING_LABELS.get(c, c) for c in corr.index]
        
        fig = px.bar(x=corr.values, y=corr.index, orientation="h",
                     labels={"x": "Impact on Overall Satisfaction", "y": ""},
                     color=corr.values, color_continuous_scale="Teal")
        fig.update_layout(height=500, xaxis_title="Correlation Strength")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Performance Gap")
        st.markdown("Current average performance across dimensions:")
        
        avg_ratings = df[RATING_COLUMNS].mean().sort_values(ascending=False)
        avg_ratings.index = [RATING_LABELS[c] for c in avg_ratings.index]
        
        st.dataframe(avg_ratings.to_frame(name="Avg Rating").round(2), use_container_width=True)

    st.divider()
    
    st.subheader("Digital Experience: Mobile vs Desktop")
    st.markdown("Comparing satisfaction levels by booking/review platform.")
    
    mobile_comp = df.groupby("via_mobile")[RATING_COLUMNS].mean().T
    mobile_comp.columns = ["Desktop", "Mobile"]
    mobile_comp.index = [RATING_LABELS[c] for c in RATING_COLUMNS]
    
    # Visual comparison instead of just a table
    mobile_melted = mobile_comp.reset_index().melt(id_vars="index", var_name="Platform", value_name="Rating")
    fig = px.bar(mobile_melted, x="index", y="Rating", color="Platform", barmode="group",
                 color_discrete_map={"Desktop": "#CBD5E0", "Mobile": "#667EEA"},
                 labels={"index": ""})
    fig.update_layout(yaxis_range=[3, 5])
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Sentiment Analysis":
    st.title("Sentiment Intelligence")
    st.markdown("_Understand the 'Why' behind the ratings using AI-powered text analysis_")

    if "sentiment_polarity" not in df.columns:
        st.error("Sentiment data not found. Please rebuild the database using `python src/data_processing.py`.")
    else:
        # Key Metrics
        avg_polarity = df["sentiment_polarity"].mean()
        positive_pct = (df["sentiment_polarity"] > 0.2).mean() * 100
        negative_pct = (df["sentiment_polarity"] < -0.2).mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Sentiment Score", f"{avg_polarity:.2f}",
                  help="Ranges from -1 (Negative) to +1 (Positive)")
        c2.metric("Positive Reviews", f"{positive_pct:.1f}%", help="Polarity > 0.2")
        c3.metric("Critical Reviews", f"{negative_pct:.1f}%", help="Polarity < -0.2")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            fig = px.histogram(df, x="sentiment_polarity", nbins=50,
                               title="Distribution of Sentiment Scores",
                               labels={"sentiment_polarity": "Sentiment Score (-1 to +1)"},
                               color_discrete_sequence=["#9F7AEA"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Sentiment vs. Rating")
            # Sample for scatter plot performance
            sample = df.sample(min(5000, len(df)))
            fig = px.strip(sample, x="rating_overall", y="sentiment_polarity",
                           title="Do High Ratings Always Mean Positive Text?",
                           labels={"rating_overall": "Stars", "sentiment_polarity": "Sentiment"},
                           color="rating_overall")
            st.plotly_chart(fig, use_container_width=True)

        # Word Clouds (using pre-computed or generating on fly - generating on fly for now)
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        st.subheader("What People are Saying")
        wc_col1, wc_col2 = st.columns(2)

        def generate_wordcloud(text_data, title, cmap):
            wc = WordCloud(width=400, height=200, background_color="#1A202C", max_words=100,
                           colormap=cmap).generate(text_data)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, color="white")
            return fig

        with st.spinner("Analyzing text patterns..."):
            with wc_col1:
                pos_text = " ".join(df[df["sentiment_polarity"] > 0.4]["text"].head(1000))
                if pos_text:
                    st.pyplot(generate_wordcloud(pos_text, "Positive Key Themes", "Greens"))
                else:
                    st.info("Not enough positive data.")

            with wc_col2:
                neg_text = " ".join(df[df["sentiment_polarity"] < -0.1]["text"].head(1000))
                if neg_text:
                    st.pyplot(generate_wordcloud(neg_text, "Negative Key Themes", "Reds"))
                else:
                    st.info("Not enough negative data.")
    
    st.info("Note: Sentiment analysis uses Natural Language Processing (TextBlob) to evaluate the emotional tone of reviews.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Hotel Performance Dashboard · v2.1")
