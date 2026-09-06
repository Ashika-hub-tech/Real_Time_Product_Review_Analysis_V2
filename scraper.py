"""
Flipkart review scraper.

Design goals
------------
* Does NOT depend on Flipkart's auto-generated CSS class names (they rotate
  every few weeks and were the reason the original scraper broke).
  Instead it identifies review cards *structurally*: a card is the smallest
  element that contains a star rating (a 1-5 digit next to a star icon) AND a
  "Certified Buyer" / "READ MORE" marker.  Known class names are still tried
  first as a fast path.
* Accepts a normal product URL (/p/) or a reviews URL (/product-reviews/) and
  normalises it.
* Stops automatically when a page yields no reviews (no hard-coded 43 pages).
* Realistic browser headers, retries with back-off, polite delay, and clear
  errors when Flipkart blocks the request (403 / 429 / captcha page).

Usage
-----
    python scraper.py "<flipkart url>" --pages 5 --out reviews.csv

or from Python:

    from scraper import scrape_reviews
    df = scrape_reviews(url, max_pages=5)
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Optional
from urllib.parse import parse_qs, urlparse, urlunparse, urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
]


def _headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
        # NOTE: no "br" here - requests can't decode brotli without an extra package
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.flipkart.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


class ScrapeError(RuntimeError):
    """Raised when Flipkart cannot be reached or actively blocks us."""


def fetch(url: str, session: requests.Session, retries: int = 3, timeout: int = 20) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=_headers(), timeout=timeout)
            if resp.status_code in (403, 429, 503):
                raise ScrapeError(
                    f"Flipkart returned HTTP {resp.status_code} (rate-limited / blocked). "
                    "Wait a few minutes, reduce --pages, or try from a different network."
                )
            resp.raise_for_status()
            html = resp.text
            if "<html" not in html[:3000].lower():
                raise ScrapeError(
                    "Response is not HTML (got binary/compressed data). "
                    f"Content-Encoding={resp.headers.get('Content-Encoding')!r}. "
                    "If it says 'br', run: pip install brotli"
                )
            if "captcha" in html.lower()[:5000] and "review" not in html.lower()[:5000]:
                raise ScrapeError("Flipkart served a CAPTCHA page instead of reviews.")
            return html
        except ScrapeError:
            raise
        except requests.RequestException as e:  # network hiccup -> retry
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise ScrapeError(f"Could not fetch {url}: {last_err}")


# --------------------------------------------------------------------------- #
# URL handling
# --------------------------------------------------------------------------- #

def normalise_url(url: str) -> str:
    """
    Turn any Flipkart product / reviews URL into a clean reviews URL with a
    `page={}` placeholder.

    https://www.flipkart.com/<slug>/p/<itm>?pid=XYZ&...
    https://www.flipkart.com/<slug>/product-reviews/<itm>?pid=XYZ&page=2
    ->  https://www.flipkart.com/<slug>/product-reviews/<itm>?pid=XYZ&page={}
    """
    url = url.strip()
    if "flipkart.com" not in url:
        raise ValueError("That doesn't look like a Flipkart URL.")
    if not url.startswith("http"):
        url = "https://" + url

    parts = urlparse(url)
    path = parts.path.replace("/p/", "/product-reviews/")
    if "/product-reviews/" not in path:
        raise ValueError("URL must contain '/p/' or '/product-reviews/'.")

    qs = parse_qs(parts.query)
    pid = qs.get("pid", [None])[0]
    if not pid:
        raise ValueError("URL is missing the 'pid' parameter.")

    query = urlencode({"pid": pid, "marketplace": "FLIPKART", "page": "{}"})
    # urlencode escapes {} -> %7B%7D, undo that for the placeholder
    query = query.replace("%7B%7D", "{}")
    return urlunparse(("https", "www.flipkart.com", path, "", query, ""))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

@dataclass
class Review:
    customer_name: str
    review_title: str
    rating: Optional[int]
    comment: str


_RATING_RE = re.compile(r"^\s*([1-5])\s*$")
_MARKER_RE = re.compile(r"(certified|verified) buyer|read more|\b(day|week|month|year)s? ago", re.I)
_NOISE_RE = re.compile(
    r"^(read more|more|(certified|verified) buyer.*|permalink|report abuse|\d+\s*(day|week|month|year)s? ago|helpful|\d+|[\d\s★]+)$",
    re.I,
)

# Fast path: class names Flipkart has used at various times. Extend freely.
KNOWN_SELECTORS = {
    "comment": ["div.ZmyHeo", "div.t-ZTKy", "div._6K-7Co"],
    "title":   ["p.z9E0IG", "p._2-N8zT"],
    "name":    ["p._2NsDsF.AwS1CA", "p._2sc7ZR._2V5EHH", "p._2sc7ZR"],
    "rating":  ["div.XQDdHH", "div._3LWZlK", "div._2wzgFH"],
}


def _text(el: Optional[Tag]) -> str:
    return el.get_text(" ", strip=True) if el else ""


def _clean_comment(txt: str) -> str:
    return re.sub(r"\s*READ MORE\s*$", "", txt, flags=re.I).strip()


_STAR_CHARS = "★☆⭐"


def _own_text(el: Tag) -> str:
    return "".join(t for t in el.find_all(string=True, recursive=False))


def _rating_value(el: Tag) -> Optional[int]:
    """Return 1-5 if this element *is* a star-rating badge, else None."""
    own = _own_text(el).replace("\xa0", " ")
    own_wo_star = re.sub(f"[{_STAR_CHARS}]", "", own)
    if not _RATING_RE.match(own_wo_star):
        return None
    has_star = (
        any(c in own for c in _STAR_CHARS)
        or el.find("img") is not None or el.find("svg") is not None
        or any(c in _STAR_CHARS for c in el.get_text())
        or "star" in " ".join(el.get("class", [])).lower()
    )
    return int(own_wo_star.strip()) if has_star else None


def _find_rating_nodes(soup: BeautifulSoup) -> list[Tag]:
    """Elements whose own text is exactly a single digit 1-5 next to a star."""
    return [el for el in soup.find_all(["div", "span", "p"]) if _rating_value(el) is not None]


def _card_for(rating_el: Tag) -> Optional[Tag]:
    """Walk up until we hit an element that also contains a review marker."""
    node = rating_el
    for _ in range(8):
        node = node.parent
        if node is None or node.name in ("body", "html"):
            return None
        if node.find(string=_MARKER_RE):
            return node
    return None


def _parse_card(card: Tag, rating_el: Tag) -> Optional[Review]:
    rating = _rating_value(rating_el)

    # Title: a <p> that shares the same immediate row as the rating.
    title = ""
    row = rating_el.parent
    if row is not None:
        p = row.find("p")
        title = _text(p)

    # Name: the <p> immediately before "Certified Buyer".
    name = ""
    cb = card.find(string=re.compile(r"(certified|verified) buyer", re.I))
    if cb is not None:
        holder = cb.find_parent("p") or cb.parent
        prev = holder.find_previous_sibling("p") if holder else None
        name = _text(prev)

    # Comment: the longest text block in the card that isn't title/name/noise.
    candidates = []
    for el in card.find_all(["div", "p", "span"]):
        if el.find(["div", "p"]):          # only leaf-ish blocks
            continue
        t = _clean_comment(_text(el))
        if not t or t in (title, name) or _NOISE_RE.match(t) or _RATING_RE.match(t):
            continue
        candidates.append(t)
    comment = max(candidates, key=len) if candidates else ""

    if not comment and not title:
        return None
    return Review(name or "Anonymous", title, rating, comment or title)


_JSON_TEXT_KEYS = ("text", "reviewText", "comment", "review", "description")
_JSON_TITLE_KEYS = ("title", "reviewTitle", "heading")
_JSON_NAME_KEYS = ("author", "authorName", "name", "userName", "reviewer")
_JSON_RATING_KEYS = ("rating", "ratingValue", "stars", "value")


def _pick(d: dict, keys: Iterable[str]):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (str, int, float)) and str(v).strip():
            return v
        if isinstance(v, dict):                    # e.g. rating: {"value": 5}
            inner = _pick(v, ("value", "text", "name"))
            if inner is not None:
                return inner
    return None


def _walk_json(obj, found: list[Review], depth: int = 0) -> None:
    if depth > 40:
        return
    if isinstance(obj, dict):
        txt, rating = _pick(obj, _JSON_TEXT_KEYS), _pick(obj, _JSON_RATING_KEYS)
        if isinstance(txt, str) and len(txt) > 10 and rating is not None:
            try:
                r = int(float(rating))
            except (TypeError, ValueError):
                r = None
            if r is not None and 1 <= r <= 5:
                found.append(Review(str(_pick(obj, _JSON_NAME_KEYS) or "Anonymous"),
                                    str(_pick(obj, _JSON_TITLE_KEYS) or ""), r, txt.strip()))
                return
        for v in obj.values():
            _walk_json(v, found, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _walk_json(v, found, depth + 1)


def parse_reviews_from_json(html: str) -> list[Review]:
    """Flipkart embeds page state as JSON inside <script> tags; mine it."""
    import json
    soup = BeautifulSoup(html, "lxml")
    found: list[Review] = []
    for sc in soup.find_all("script"):
        body = sc.string or sc.get_text()
        if not body or "rating" not in body.lower():
            continue
        # try whole body (application/json) or the first {...} after an '='
        cands = [body.strip()]
        m = re.search(r"=\s*(\{.*\})\s*;?\s*$", body.strip(), re.S)
        if m:
            cands.append(m.group(1))
        for c in cands:
            try:
                _walk_json(json.loads(c), found)
                break
            except (ValueError, RecursionError):
                continue
    # dedupe
    out, seen = [], set()
    for r in found:
        if r.comment not in seen:
            seen.add(r.comment); out.append(r)
    return out


def parse_reviews(html: str) -> list[Review]:
    reviews = _parse_reviews_html(html)
    if not reviews:
        reviews = parse_reviews_from_json(html)
    return reviews


def _parse_reviews_html(html: str) -> list[Review]:
    soup = BeautifulSoup(html, "lxml")

    # --- fast path with known class names -------------------------------- #
    def first_hit(kind: str) -> list[Tag]:
        for sel in KNOWN_SELECTORS[kind]:
            hits = soup.select(sel)
            if hits:
                return hits
        return []

    comments = first_hit("comment")
    titles = first_hit("title")
    names = first_hit("name")
    ratings = first_hit("rating")
    if comments and len(comments) == len(titles) == len(ratings):
        reviews = []
        for i, c in enumerate(comments):
            reviews.append(Review(
                _text(names[i]) if i < len(names) else "Anonymous",
                _text(titles[i]),
                _rating_value(ratings[i]),
                _clean_comment(_text(c)),
            ))
        if reviews:
            return reviews

    # --- structural fallback (class-name independent) -------------------- #
    reviews, seen = [], set()
    for rating_el in _find_rating_nodes(soup):
        card = _card_for(rating_el)
        if card is None or id(card) in seen:
            continue
        seen.add(id(card))
        rv = _parse_card(card, rating_el)
        if rv:
            reviews.append(rv)
    return reviews


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def scrape_reviews(
    url: str,
    max_pages: int = 10,
    delay: float = 1.5,
    progress: Optional[Callable[[int, int, int], None]] = None,
) -> pd.DataFrame:
    """
    Scrape up to `max_pages` pages of reviews.
    `progress(page, max_pages, total_reviews_so_far)` is called after each page.
    Returns a DataFrame with columns: Customer Name, Review Title, Rating, Comment.
    """
    template = normalise_url(url)
    session = requests.Session()
    all_reviews: list[Review] = []

    for page in range(1, max_pages + 1):
        html = fetch(template.format(page), session)
        page_reviews = parse_reviews(html)
        if not page_reviews:
            if page == 1:                        # keep evidence for debugging
                with open("debug_flipkart_page1.html", "w", encoding="utf-8") as f:
                    f.write(html)
            break                                # ran out of pages
        all_reviews.extend(page_reviews)
        if progress:
            progress(page, max_pages, len(all_reviews))
        time.sleep(delay + random.random())      # be polite

    df = pd.DataFrame([asdict(r) for r in all_reviews])
    if df.empty:
        raise ScrapeError(
            "No reviews found. Either the product has no reviews, or Flipkart's "
            "page layout changed in a way the parser can't recognise. The raw page was "
            "saved to debug_flipkart_page1.html — share it to get the parser updated."
        )
    df.columns = ["Customer Name", "Review Title", "Rating", "Comment"]
    df = df.drop_duplicates(subset=["Comment"]).reset_index(drop=True)
    return df


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Scrape Flipkart reviews to CSV")
    ap.add_argument("url")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--out", default="reviews.csv")
    a = ap.parse_args()
    try:
        df = scrape_reviews(a.url, a.pages,
                            progress=lambda p, m, n: print(f"page {p}/{m}: {n} reviews", file=sys.stderr))
    except (ScrapeError, ValueError) as e:
        sys.exit(f"error: {e}")
    df.to_csv(a.out, index=False)
    print(f"saved {len(df)} reviews -> {a.out}")


if __name__ == "__main__":
    _cli()
