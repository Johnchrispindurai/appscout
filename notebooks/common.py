"""Shared by every notebook and the web app: paths, cleaning, and the results file.

Every number the app, the report or the study guide shows is written by `record()` into
results/results.json, from the notebook that computed it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results" / "results.json"
SEED = 2026

# An app "succeeded" when it reached 1,000,000 installs: the bucket where the Play Store
# starts calling an app popular, and close to a 40/60 split of the 2018 data.
SUCCESS_INSTALLS = 1_000_000

PALETTE = {"main": "#2f7fc1", "accent": "#e8833a", "good": "#2e9e6b", "bad": "#c94f4f",
           "muted": "#9aa3ad", "ink": "#1f2a44"}


def style() -> None:
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "font.size": 10, "axes.titleweight": "bold"})


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")


def record(key: str, value) -> None:
    """Write one result under a dotted key, keeping everything else in the file."""
    RESULTS.parent.mkdir(exist_ok=True)
    res = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    node = res
    *path, last = key.split(".")
    for p in path:
        node = node.setdefault(p, {})
    node[last] = _no_nan(json.loads(json.dumps(value, default=_plain)))
    RESULTS.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def _no_nan(o):
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: _no_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_no_nan(v) for v in o]
    return o


def _plain(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return round(float(o), 4)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(type(o))


# ---------------------------------------------------------------- the 2018 Kaggle snapshot

def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA / "googleplaystore.csv")


def size_mb(s: str) -> float:
    """'19M' -> 19.0, '201k' -> 0.196..., 'Varies with device' -> NaN."""
    if isinstance(s, str) and s.endswith("M"):
        return float(s[:-1])
    if isinstance(s, str) and s.endswith("k"):
        return float(s[:-1]) / 1024
    return np.nan


def min_android(s) -> float:
    """'4.0.3 and up' -> 4.0, '7.0 - 7.1.1' -> 7.0, 'Varies with device' -> NaN."""
    m = re.match(r"(\d+)\.(\d+)", str(s))
    return float(f"{m[1]}.{m[2]}") if m else np.nan


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The 2018 app table, one row per app, with numeric columns. Returns (apps, log of what was done)."""
    log = {"raw_rows": len(raw)}
    # One row has every value shifted one column left (Category '1.9', Installs 'Free').
    df = raw[raw["Category"] != "1.9"].copy()
    log["shifted_rows_dropped"] = len(raw) - len(df)
    df["installs"] = df["Installs"].str.replace(r"[+,]", "", regex=True).astype(int)
    df["reviews"] = df["Reviews"].astype(int)
    df["price"] = df["Price"].str.replace("$", "", regex=False).astype(float)
    df["size_mb"] = df["Size"].map(size_mb)
    df["android"] = df["Android Ver"].map(min_android)
    df["updated"] = pd.to_datetime(df["Last Updated"], format="%B %d, %Y")
    df["rating"] = df["Rating"]
    df["category"] = df["Category"]
    df["genre"] = df["Genres"].str.split(";").str[0]
    df["n_genres"] = df["Genres"].str.count(";") + 1
    df["paid"] = (df["Type"] == "Paid").astype(int)
    df["content"] = df["Content Rating"]
    # The same app is listed several times: re-scrapes (more reviews later) and apps shown
    # under two categories (often FAMILY and GAME). Keep the row with the most reviews.
    log["duplicate_rows"] = int(df.duplicated("App").sum())
    log["apps_in_two_categories"] = int((df.groupby("App")["category"].nunique() > 1).sum())
    df = df.sort_values("reviews", ascending=False).drop_duplicates("App").sort_index()
    df = df.rename(columns={"App": "app"})
    df["log_installs"] = np.log10(df["installs"].clip(lower=1))
    df["success"] = (df["installs"] >= SUCCESS_INSTALLS).astype(int)
    log["apps"] = len(df)
    log["missing_rating"] = int(df["rating"].isna().sum())
    log["size_varies"] = int(df["size_mb"].isna().sum())
    cols = ["app", "category", "genre", "n_genres", "rating", "reviews", "installs", "log_installs", "success",
            "size_mb", "paid", "price", "content", "android", "updated"]
    return df[cols].reset_index(drop=True), log


def load_apps() -> pd.DataFrame:
    return clean(load_raw())[0]


def split(apps: pd.DataFrame) -> np.ndarray:
    """'train' / 'val' / 'test' for every app, 60/20/20, stratified by category, fixed seed.
    Every notebook uses this split: models are tuned on val and scored once on test."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(apps))
    train, rest = train_test_split(idx, test_size=0.4, random_state=SEED, stratify=apps["category"])
    val, test = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=apps["category"].iloc[rest])
    out = np.empty(len(apps), dtype=object)
    out[train], out[val], out[test] = "train", "val", "test"
    return out


def load_reviews() -> pd.DataFrame:
    """User reviews with sentiment, rows without a review dropped."""
    r = pd.read_csv(DATA / "googleplaystore_user_reviews.csv").dropna(subset=["Translated_Review", "Sentiment"])
    return r.rename(columns={"App": "app", "Translated_Review": "review", "Sentiment": "sentiment",
                             "Sentiment_Polarity": "polarity", "Sentiment_Subjectivity": "subjectivity"})


# ---------------------------------------------------------------- the 2026 scrape

def category_of(genre_id: str) -> str:
    """The 2026 store splits games into GAME_PUZZLE, GAME_RACING...; 2018 had one GAME."""
    return "GAME" if str(genre_id).startswith("GAME_") else str(genre_id)


def load_2026() -> pd.DataFrame:
    df = pd.read_csv(DATA / "playstore_2026.csv")
    df["category"] = df["genreId"].map(category_of)
    df["installs"] = df["minInstalls"].fillna(0).astype(int)
    df["log_installs"] = np.log10(df["installs"].clip(lower=1))
    df["success"] = (df["installs"] >= SUCCESS_INSTALLS).astype(int)
    df["paid"] = (~df["free"].astype(bool)).astype(int)
    return df.rename(columns={"title": "app", "score": "rating", "ratings": "n_ratings"})


def demo() -> None:
    assert size_mb("19M") == 19.0 and abs(size_mb("512k") - 0.5) < 1e-9 and np.isnan(size_mb("Varies with device"))
    assert min_android("4.0.3 and up") == 4.0 and min_android("7.0 - 7.1.1") == 7.0 and np.isnan(min_android("Varies"))
    assert category_of("GAME_PUZZLE") == "GAME" and category_of("FINANCE") == "FINANCE"
    raw = pd.DataFrame({"App": ["a", "a", "b", "x"], "Category": ["FAMILY", "GAME", "TOOLS", "1.9"],
                        "Rating": [4.0, 4.1, np.nan, 19.0], "Reviews": ["10", "20", "0", "3.0M"],
                        "Size": ["1M", "1M", "100k", "1,000+"], "Installs": ["1,000+", "1,000,000+", "0", "Free"],
                        "Type": ["Free", "Free", "Paid", "0"], "Price": ["0", "0", "$2.99", "Everyone"],
                        "Content Rating": ["Everyone"] * 3 + [np.nan], "Genres": ["Puzzle;Brain Games", "Puzzle", "Tools", "x"],
                        "Last Updated": ["January 7, 2018"] * 3 + ["1.0.19"], "Current Ver": ["1"] * 4,
                        "Android Ver": ["4.0.3 and up"] * 3 + [np.nan]})
    apps, log = clean(raw)
    assert log["shifted_rows_dropped"] == 1 and log["apps"] == 2 and log["apps_in_two_categories"] == 1
    a = apps.set_index("app")
    assert a.loc["a", "category"] == "GAME" and a.loc["a", "success"] == 1 and a.loc["b", "price"] == 2.99
    assert a.loc["b", "log_installs"] == 0.0
    sp = split(pd.DataFrame({"category": ["A"] * 10 + ["B"] * 10}))
    assert (sp == "train").sum() == 12 and (sp == "val").sum() == 4 and (sp == "test").sum() == 4
    print("common ok")


if __name__ == "__main__":
    demo()
