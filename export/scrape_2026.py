"""Scrape a fresh sample of the Google Play Store, to set beside the 2018 Kaggle snapshot.

For every category, two search phrases (the kind of thing a user types when looking for an
app); every hit is then looked up for its full details. US store, English.

    python export/scrape_2026.py          # writes data/playstore_2026.csv and data/scrape_log.json

Search results are ranked by Google, so the sample leans toward popular apps. That bias
is stated wherever the 2026 data is used.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google_play_scraper import app, search

DATA = Path(__file__).resolve().parents[1] / "data"

QUERIES = {
    "ART_AND_DESIGN": ["drawing app", "logo maker"], "AUTO_AND_VEHICLES": ["car maintenance", "used cars"],
    "BEAUTY": ["makeup tutorial", "hairstyle"], "BOOKS_AND_REFERENCE": ["ebook reader", "dictionary"],
    "BUSINESS": ["invoice maker", "crm"], "COMICS": ["comics reader", "manga"],
    "COMMUNICATION": ["messenger", "video call"], "DATING": ["dating", "meet singles"],
    "EDUCATION": ["learn english", "math learning"], "ENTERTAINMENT": ["streaming movies", "fun quiz"],
    "EVENTS": ["event tickets", "event planner"], "FINANCE": ["budget planner", "stock trading"],
    "FOOD_AND_DRINK": ["recipes", "food delivery"], "HEALTH_AND_FITNESS": ["calorie counter", "home workout"],
    "HOUSE_AND_HOME": ["home design", "smart home"], "LIBRARIES_AND_DEMO": ["sdk demo", "sample app"],
    "LIFESTYLE": ["horoscope", "habit tracker"], "GAME": ["puzzle game", "racing game"],
    "FAMILY": ["kids learning", "baby games"], "MEDICAL": ["medicine reminder", "doctor appointment"],
    "SOCIAL": ["social network", "community chat"], "SHOPPING": ["online shopping", "coupons"],
    "PHOTOGRAPHY": ["photo editor", "camera filter"], "SPORTS": ["live scores", "football news"],
    "TRAVEL_AND_LOCAL": ["hotel booking", "travel planner"], "TOOLS": ["file manager", "vpn"],
    "PERSONALIZATION": ["wallpaper", "launcher"], "PRODUCTIVITY": ["to do list", "notes"],
    "PARENTING": ["pregnancy tracker", "baby tracker"], "WEATHER": ["weather forecast", "weather radar"],
    "VIDEO_PLAYERS": ["video player", "video editor"], "NEWS_AND_MAGAZINES": ["news", "magazine"],
    "MAPS_AND_NAVIGATION": ["gps navigation", "offline maps"],
}
FIELDS = ["appId", "title", "genre", "genreId", "score", "ratings", "reviews", "minInstalls", "realInstalls",
          "free", "price", "contentRating", "released", "lastUpdatedOn", "summary", "developer"]


def details(app_id):
    for attempt in range(3):
        try:
            d = app(app_id, lang="en", country="us")
            return {k: d.get(k) for k in FIELDS}
        except Exception:  # ponytail: 3 tries then skip; the log counts what was lost
            time.sleep(2 * (attempt + 1))
    return None


def main():
    hits = {}
    for cat, phrases in QUERIES.items():
        for q in phrases:
            for h in search(q, n_hits=30, lang="en", country="us"):
                hits.setdefault(h["appId"], (cat, q))
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(details, hits))
    df = pd.DataFrame([dict(r, query_category=hits[r["appId"]][0], query=hits[r["appId"]][1]) for r in rows if r])
    df.to_csv(DATA / "playstore_2026.csv", index=False, encoding="utf-8")
    log = {"scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "store": "US, English",
           "queries": sum(map(len, QUERIES.values())), "search_hits": len(hits), "apps_with_details": len(df),
           "failed_lookups": len(hits) - len(df), "package": "google-play-scraper"}
    (DATA / "scrape_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(log)


if __name__ == "__main__":
    main()
