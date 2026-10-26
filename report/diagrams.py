"""Draw the two block diagrams for the report: system architecture and the idea-evaluation flow.

    python report/diagrams.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "assets"
INK, MUTED = "#15212c", "#4b5864"
FILL = {"source": "#e8f1fb", "store": "#f4f6f7", "work": "#fdeee4", "out": "#e6f4ee"}


def box(ax, x, y, w, h, title, lines=(), kind="work"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[kind], ec=INK, lw=1.1))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.6 - i * 0.3, line, ha="center", va="top", fontsize=8.6, color=MUTED)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.2, color=INK))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
    box(ax, 0.2, 4.6, 2.6, 1.8, "2018 snapshot", ["9,659 apps (Kaggle,", "scraped Aug 2018)", "CC BY 3.0"], "source")
    box(ax, 0.2, 2.5, 2.6, 1.8, "User reviews", ["37,427 reviews", "with sentiment", "865 apps"], "source")
    box(ax, 0.2, 0.4, 2.6, 1.8, "2026 scrape", ["google-play-scraper", "66 searches,", "1,668 apps"], "source")
    box(ax, 3.5, 2.3, 2.4, 2.2, "Cleaning", ["shifted row, duplicates", "text to numbers", "log installs", "(common.py)"], "store")
    box(ax, 6.5, 2.3, 2.3, 2.2, "Viability engine", ["TF-IDF of names", "category model", "neighbourhood (K=25)", "success model"], "work")
    box(ax, 6.5, 4.8, 2.3, 1.6, "AppScout web app", ["Streamlit: idea in,", "verdict + charts out"], "out")
    box(ax, 9.4, 2.0, 2.4, 2.8, "Analysis notebooks", ["1 data quality", "2 explore", "3 regression", "4 classification",
                                                        "5 clustering", "6 tests, fairness", "7 recommender"], "work")
    box(ax, 9.4, 0.2, 2.4, 1.4, "results.json", ["every number,", "written by code"], "store")
    box(ax, 6.5, 0.2, 2.3, 1.4, "Outputs", ["report, figures,", "app captions"], "out")
    for y in (5.5, 3.4, 1.3):
        arrow(ax, (2.8, y), (3.5, 3.4))
    arrow(ax, (5.9, 3.4), (6.5, 3.4)); arrow(ax, (8.8, 3.4), (9.4, 3.4))
    arrow(ax, (7.65, 4.5), (7.65, 4.8)); arrow(ax, (10.6, 2.0), (10.6, 1.6)); arrow(ax, (9.4, 0.9), (8.8, 0.9))
    fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight", facecolor="white")


def flow():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.axis("off")
    steps = [("Idea text", ["one sentence", "free/paid, price,", "content rating"]), ("Vectorise", ["noise words out", "char 3-5-grams", "TF-IDF"]),
             ("Place it", ["category: logistic", "regression (top 3)", "user can override"]),
             ("Neighbourhood", ["25 most similar", "app names", "installs, crowding"]),
             ("Score", ["P(1M+ installs)", "percentile = score", "exact drivers"]),
             ("Suggest", ["better niches", "complaint topics", "winners' words"])]
    w, gap = 1.75, 0.27
    for i, (t, lines) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        box(ax, x, 0.6, w, 2.2, t, lines, "work" if i in (3, 4) else "store")
        if i:
            arrow(ax, (x - gap, 1.7), (x, 1.7))
    fig.savefig(OUT / "flow.png", dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    architecture(); flow()
    print("diagrams written")
