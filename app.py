"""
Real-Time Product Review Analysis  –  Streamlit app

Tabs
  1. Scrape Flipkart   – paste a product URL, scrape N pages, analyse, download CSV
  2. Upload CSV        – any CSV with a `Comment` column
  3. Analyse Text      – single review -> 28-class emotion + sentiment
Run:  streamlit run app.py
"""
from __future__ import annotations

import io
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

from scraper import ScrapeError, scrape_reviews

st.set_page_config(page_title="Product Review Analysis", page_icon="📊", layout="wide")
st.title("📊 Real-Time Product Review Analysis")
st.caption("Flipkart scraping · GoEmotions emotion detection · sentiment classification · brand-perception dashboard")

# --------------------------------------------------------------------------- #
# Models (cached once per server process, not per rerun)
# --------------------------------------------------------------------------- #

EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

POSITIVE = {"admiration", "amusement", "approval", "caring", "desire", "excitement",
            "gratitude", "joy", "love", "optimism", "pride", "relief"}
NEGATIVE = {"anger", "annoyance", "disappointment", "disapproval", "disgust",
            "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"}


@st.cache_resource(show_spinner="Loading emotion model (first run downloads ~500 MB)…")
def emotion_pipe():
    from transformers import pipeline
    return pipeline("text-classification", model=EMOTION_MODEL, device=-1, truncation=True)


@st.cache_resource(show_spinner="Loading sentiment model…")
def sentiment_pipe():
    from transformers import pipeline
    return pipeline("text-classification", model=SENTIMENT_MODEL, device=-1, truncation=True)


def to_category(emotion: str) -> str:
    if emotion in POSITIVE:
        return "Positive"
    if emotion in NEGATIVE:
        return "Negative"
    return "Neutral"


@st.cache_data(show_spinner=False)
def analyse(comments: tuple[str, ...]) -> pd.DataFrame:
    """Batched emotion + sentiment inference. Tuple input so it's hashable for caching."""
    texts = [str(c)[:1500] for c in comments]
    emo = emotion_pipe()(texts, batch_size=16)
    sen = sentiment_pipe()(texts, batch_size=16)
    return pd.DataFrame({
        "Comment": comments,
        "Emotion": [e["label"] for e in emo],
        "Emotion Score": [round(e["score"], 3) for e in emo],
        "Emotion Category": [to_category(e["label"]) for e in emo],
        "Sentiment": [s["label"].capitalize() for s in sen],
        "Sentiment Score": [round(s["score"], 3) for s in sen],
    })


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #

CAT_COLOURS = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}


def dashboard(df: pd.DataFrame, title: str) -> None:
    st.subheader(f"Results · {title}")

    n = len(df)
    pos = (df["Emotion Category"] == "Positive").mean() * 100
    neg = (df["Emotion Category"] == "Negative").mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews analysed", n)
    c2.metric("Positive", f"{pos:.0f}%")
    c3.metric("Negative", f"{neg:.0f}%")
    if "Rating" in df.columns and df["Rating"].notna().any():
        c4.metric("Avg rating", f"{pd.to_numeric(df['Rating'], errors='coerce').mean():.2f} ★")
    else:
        c4.metric("Neutral", f"{100 - pos - neg:.0f}%")

    cat = df["Emotion Category"].value_counts().rename_axis("Category").reset_index(name="Count")
    emo = df["Emotion"].value_counts().head(10).rename_axis("Emotion").reset_index(name="Count")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.pie(cat, values="Count", names="Category", title="Overall perception",
                               color="Category", color_discrete_map=CAT_COLOURS), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(emo, x="Emotion", y="Count", title="Top emotions",
                               color="Count", color_continuous_scale="Viridis"), use_container_width=True)

    if "Rating" in df.columns and df["Rating"].notna().any():
        tmp = df.assign(Rating=pd.to_numeric(df["Rating"], errors="coerce")).dropna(subset=["Rating"])
        ct = pd.crosstab(tmp["Rating"], tmp["Emotion Category"]).reset_index().melt(
            id_vars="Rating", var_name="Category", value_name="Count")
        st.plotly_chart(px.bar(ct, x="Rating", y="Count", color="Category", barmode="stack",
                               title="Star rating vs detected emotion (model sanity check)",
                               color_discrete_map=CAT_COLOURS), use_container_width=True)

    with st.expander("Per-review table"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button("⬇ Download analysed CSV", df.to_csv(index=False).encode(),
                       file_name=f"{title}_analysed.csv", mime="text/csv")


def run_analysis(df: pd.DataFrame, title: str) -> None:
    if "Comment" not in df.columns:
        st.error(f"CSV needs a 'Comment' column. Found: {list(df.columns)}")
        return
    df = df.dropna(subset=["Comment"])
    df = df[df["Comment"].astype(str).str.strip() != ""]
    if df.empty:
        st.warning("No comments to analyse.")
        return
    with st.spinner(f"Analysing {len(df)} reviews…"):
        res = analyse(tuple(df["Comment"].astype(str)))
    merged = df.reset_index(drop=True).drop(columns=["Comment"]).join(res)
    dashboard(merged, title)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

tab_scrape, tab_csv, tab_text = st.tabs(["🔍 Scrape Flipkart", "📁 Upload CSV", "✍ Analyse Text"])

with tab_scrape:
    st.markdown("Paste any Flipkart **product** or **reviews** URL. The scraper follows pagination "
                "and stops automatically when reviews run out.")
    col1, col2 = st.columns([3, 1])
    url = col1.text_input("Flipkart URL",
                          placeholder="https://www.flipkart.com/<product>/p/<itm...>?pid=...")
    pages = col2.slider("Max pages", 1, 30, 5, help="~10 reviews per page")
    name = st.text_input("Product name (used for the CSV filename)", value="product")

    if st.button("Scrape & Analyse", type="primary"):
        if not url:
            st.warning("Enter a URL first.")
        else:
            bar = st.progress(0, "Starting…")
            try:
                df = scrape_reviews(
                    url, max_pages=pages,
                    progress=lambda p, m, n: bar.progress(p / m, f"Page {p}/{m} · {n} reviews"),
                )
                bar.progress(1.0, f"Done · {len(df)} reviews")
                df.to_csv(f"{name}_reviews.csv", index=False)
                st.session_state["scraped"] = (df, name)
            except (ScrapeError, ValueError) as e:
                bar.empty()
                st.error(str(e))
                st.info("Flipkart sometimes rate-limits datacentre / cloud IPs. Running locally "
                        "usually works; otherwise use the **Upload CSV** tab with `reviews.csv` "
                        "to try the analysis.")

    if "scraped" in st.session_state:
        df, name = st.session_state["scraped"]
        run_analysis(df, name)

with tab_csv:
    up = st.file_uploader("CSV with a `Comment` column (Rating column optional)", type="csv")
    if up is not None:
        raw = up.getvalue()
        df = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            st.error("Could not decode the file.")
        else:
            st.write(f"Loaded {len(df)} rows.")
            if st.button("Analyse CSV", type="primary"):
                run_analysis(df, up.name.rsplit(".", 1)[0])

with tab_text:
    txt = st.text_area("Type or paste a review", height=160,
                       placeholder="The battery drains in 3 hours and support never replied…")
    if st.button("Analyse", type="primary"):
        if not txt.strip():
            st.warning("Enter some text.")
        else:
            res = analyse((txt.strip(),)).iloc[0]
            a, b = st.columns(2)
            a.metric("Emotion", f"{res['Emotion']} ({res['Emotion Category']})", f"{res['Emotion Score']:.0%} conf.")
            b.metric("Sentiment", res["Sentiment"], f"{res['Sentiment Score']:.0%} conf.")
            top = emotion_pipe()(txt.strip()[:1500], top_k=5)
            st.plotly_chart(px.bar(pd.DataFrame(top), x="label", y="score", title="Top-5 emotions"),
                            use_container_width=True)
