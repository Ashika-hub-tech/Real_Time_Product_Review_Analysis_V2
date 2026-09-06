"""Diagnostic: fetch page 1 of a Flipkart reviews URL and report what the parser sees.
Usage: python debug_scrape.py "<flipkart url>"
"""
import sys, re, requests
from scraper import normalise_url, fetch, _parse_reviews_html, parse_reviews_from_json, _headers

url = normalise_url(sys.argv[1]).format(1)
print("Fetching:", url)
html = fetch(url, requests.Session())
open("debug_flipkart_page1.html", "w", encoding="utf-8").write(html)
low = html.lower()
print(f"HTML size: {len(html):,} chars")
print("mentions 'verified buyer':", low.count("verified buyer"), "| 'certified buyer':", low.count("certified buyer"))
print("mentions 'read more':", low.count("read more"), "| '★':", html.count("★"))
print("has __INITIAL_STATE__:", "__initial_state__" in low, "| script tags:", low.count("<script"))
print("HTML parser found:", len(_parse_reviews_html(html)), "reviews")
print("JSON parser found:", len(parse_reviews_from_json(html)), "reviews")
print("\nSaved raw page to debug_flipkart_page1.html")
