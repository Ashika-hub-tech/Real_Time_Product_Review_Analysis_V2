# Real-Time Product Review Analysis

Streamlit dashboard that scrapes Flipkart product reviews, detects fine-grained
emotions (28 GoEmotions classes) and sentiment with HuggingFace Transformers,
and visualises brand perception with interactive Plotly charts.

## Features

- **Resilient Flipkart scraper** – identifies review cards *structurally* (rating + "Certified Buyer" marker) instead of relying on Flipkart's rotating auto-generated CSS classes; follows pagination, stops when reviews run out, retries with back-off, and reports rate-limiting clearly. Works as a CLI or a library.
- **Emotion detection** – `SamLowe/roberta-base-go_emotions`, batched CPU inference, mapped to Positive / Negative / Neutral.
- **Sentiment classification** – `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- **Dashboard** – KPI tiles, perception pie, top-emotion bar, and a *rating-vs-emotion* chart that acts as a sanity check on the model.
- **CSV upload & download** – analyse any CSV with a `Comment` column; export the annotated results.
- **Single-text analysis** – top-5 emotions + sentiment for any review.
- **Unit tests** for the parser and URL normaliser (`pytest`).

## Quick start

```bash
git clone https://github.com/Ashika-hub-tech/REAL_TIME_PRODUCT_REVIEW_ANALYSIS.git
cd REAL_TIME_PRODUCT_REVIEW_ANALYSIS
python -m venv env && source env/bin/activate      # env\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

The two models (~1 GB) download on first use and are cached by HuggingFace.

### Scraper CLI

```bash
python scraper.py "https://www.flipkart.com/<product>/p/<itm...>?pid=..." --pages 5 --out reviews.csv
```

### Tests

```bash
pytest tests/
```

## Project structure

```
app.py               Streamlit UI (3 tabs: scrape / upload / text)
scraper.py           Flipkart scraper – library + CLI
tests/test_parser.py Parser & URL tests with an HTML fixture
sample_reviews.csv   Small dataset for trying the Upload tab
requirements.txt
```

## How the scraper stays alive when Flipkart changes its HTML

Flipkart's class names (`_2sc7ZR`, `t-ZTKy`, `ZmyHeo` …) are build artefacts that
change every few weeks, which is why most tutorial scrapers stop working. `scraper.py`
tries known class names first (fast path) and otherwise finds each review by its
*shape*: the smallest element containing a star rating (a lone digit 1-5 beside a
star icon) and a "Certified Buyer" / "READ MORE" marker. Title, comment and reviewer
name are then extracted relative to that card.

## Limitations

- Flipkart rate-limits aggressively from cloud/datacentre IPs (HTTP 403/429). Run locally, keep `--pages` small, and expect a 1–3 s polite delay per page.
- Reviews are in English/Hinglish; the models are English-only, so code-mixed text is classified less reliably.
- Emotion → Positive/Negative/Neutral mapping is heuristic; the rating-vs-emotion chart helps you judge it per product.

## Tech stack

| Component      | Library                                       |
| -------------- | --------------------------------------------- |
| UI             | Streamlit                                     |
| NLP            | HuggingFace Transformers (RoBERTa), PyTorch   |
| Visualisation  | Plotly Express                                |
| Scraping       | requests, BeautifulSoup4, lxml                |
| Data           | pandas                                        |
| Testing        | pytest                                        |
