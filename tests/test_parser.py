"""Run:  python -m pytest tests/  (or  python tests/test_parser.py)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scraper import parse_reviews, normalise_url

# Mimics Flipkart's review-card DOM, but with random class names so the test
# only passes if the class-independent structural parser works.
FIXTURE = """
<html><body>
<div class="qwerty">
  <div class="row">
    <div class="abc123">5<img src="star.svg"></div>
    <p class="t1t1">Terrific purchase</p>
  </div>
  <div class="row"><div class="c0mm"><div><div>Battery lasts two days and the camera is superb.</div></div><span>READ MORE</span></div></div>
  <div class="row"><div class="who"><p class="nm">Rahul Sharma</p><p>Certified Buyer, Chennai</p><p>3 months ago</p></div></div>
  <div class="row"><span>12</span><span>Report Abuse</span></div>
</div>
<div class="qwerty">
  <div class="row">
    <div class="abc123">2<img src="star.svg"></div>
    <p class="t1t1">Not satisfied</p>
  </div>
  <div class="row"><div class="c0mm"><div><div>Heating issue while gaming. Disappointed.</div></div></div></div>
  <div class="row"><div class="who"><p class="nm">Priya</p><p>Certified Buyer, Pune</p><p>1 month ago</p></div></div>
</div>
</body></html>
"""

def test_structural_parse():
    rv = parse_reviews(FIXTURE)
    assert len(rv) == 2
    assert rv[0].rating == 5 and rv[0].customer_name == "Rahul Sharma"
    assert rv[0].review_title == "Terrific purchase"
    assert rv[0].comment == "Battery lasts two days and the camera is superb."
    assert rv[1].rating == 2 and rv[1].comment.startswith("Heating issue")

def test_known_class_fast_path():
    html = FIXTURE.replace("abc123", "XQDdHH").replace("t1t1", "z9E0IG") \
                  .replace("c0mm", "ZmyHeo").replace('class="nm"', 'class="_2NsDsF AwS1CA"')
    rv = parse_reviews(html)
    assert [r.rating for r in rv] == [5, 2]
    assert rv[1].customer_name == "Priya"

def test_verified_buyer_text_star():
    html = FIXTURE.replace("Certified Buyer", "Verified Buyer") \
                  .replace('5<img src="star.svg">', '5 ★').replace('2<img src="star.svg">', '2★')
    rv = parse_reviews(html)
    assert [r.rating for r in rv] == [5, 2]
    assert rv[0].customer_name == "Rahul Sharma"
    assert rv[0].comment == "Battery lasts two days and the camera is superb."

def test_json_fallback():
    html = """<html><body><div>nothing here</div>
    <script>window.__INITIAL_STATE__ = {"page":{"data":[{"widget":{"reviews":[
      {"id":1,"rating":{"value":4},"title":"Nice","text":"Solid phone for the price, screen is bright.","author":"Kavya"},
      {"id":2,"rating":5,"reviewTitle":"Wow","reviewText":"Exceeded expectations in every way.","authorName":"Dev"}
    ]}}]}};</script></body></html>"""
    rv = parse_reviews(html)
    assert [(r.rating, r.customer_name) for r in rv] == [(4, "Kavya"), (5, "Dev")]

def test_normalise_url():
    p = "https://www.flipkart.com/motorola-g84-5g/p/itmed938e33ffdf5?pid=MOBGQFX672GDDQAQ&lid=XYZ&marketplace=FLIPKART"
    r = normalise_url(p)
    assert r == "https://www.flipkart.com/motorola-g84-5g/product-reviews/itmed938e33ffdf5?pid=MOBGQFX672GDDQAQ&marketplace=FLIPKART&page={}"
    assert normalise_url(r.format(3)) == r          # already a reviews URL
    assert r.format(2).endswith("page=2")

if __name__ == "__main__":
    for fn in (test_structural_parse, test_known_class_fast_path, test_verified_buyer_text_star, test_json_fallback, test_normalise_url):
        fn(); print("PASS", fn.__name__)
