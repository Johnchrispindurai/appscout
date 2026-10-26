"""Everything AppScout shows for one evaluated idea: the listing card, the charts, the suggestions
and the tables. Shared by the web app (app.py) and the Colab notebook, so both show the same thing."""

from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd

import engine as E  # the caller puts notebooks/ on sys.path first

INK, MUTED, LINE = "#000000", "#4B5160", "#C9D6E8"
SKY, BLUE, LILAC = "#D8E9FF", "#2D6BFF", "#E7D9FF"
BANDS = [("High risk", "#F0442E"), ("Tough market", "#FF9F1C"), ("Promising", "#2D6BFF"), ("Strong opportunity", "#1FB26B")]
VERDICT_COLOUR = dict(BANDS)
CONTENT = ["Everyone", "Everyone 10+", "Teen", "Mature 17+"]
nice = lambda c: c.replace("_", " ").title().replace(" And ", " & ")
esc = html.escape

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Archivo:wght@400;500;600;700;800&display=swap');
.as-root {{ font-family: 'Archivo', 'Segoe UI', sans-serif; color: {INK}; }}

/* the verdict: the idea shown as the store listing it would get */
.listing {{ background: {SKY}; border: 2.5px solid #000; border-radius: 32px; padding: 1.8rem 2rem 1.6rem; margin: 1.8rem 0 1rem; }}
.listing .l-top {{ display: flex; gap: 1.2rem; align-items: center; flex-wrap: wrap; }}
.listing .icon {{ flex: 0 0 84px; height: 84px; border-radius: 24px; background: var(--v); border: 2.5px solid #000; color: #000;
  display: grid; place-items: center; font-family: 'Anton', sans-serif; font-size: 2.3rem; box-shadow: 4px 4px 0 #000; }}
.listing .l-name {{ flex: 1 1 320px; }}
.listing .title {{ font-weight: 800; font-size: clamp(1.35rem, 2.5vw, 1.9rem); line-height: 1.1; letter-spacing: -0.01em; }}
.listing .cat {{ margin-top: .45rem; display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; color: {MUTED}; font-size: .95rem; }}
.listing .pill {{ border: 2px solid #000; border-radius: 999px; padding: .1rem .75rem; background: #fff; color: #000; font-weight: 700; }}
.listing .verdict {{ background: var(--v); border: 2.5px solid #000; border-radius: 999px; padding: .55rem 1.3rem; font-weight: 800;
  font-size: 1.15rem; transform: rotate(-4deg); box-shadow: 4px 4px 0 #000; white-space: nowrap; }}
.listing .l-main {{ display: grid; grid-template-columns: auto 1fr; gap: 2.2rem; align-items: center; margin: 1.4rem 0 1.6rem; }}
.listing .score {{ display: flex; align-items: flex-end; gap: .6rem; }}
.listing .num {{ font-family: 'Anton', sans-serif; font-size: clamp(6rem, 12vw, 9.5rem); line-height: .82; }}
.listing .den {{ font-weight: 700; font-size: 1rem; line-height: 1.25; padding-bottom: .4rem; }}
.listing .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); background: #fff; border: 2.5px solid #000; border-radius: 24px; overflow: hidden; }}
.listing .stat {{ padding: .9rem 1.2rem; border-right: 2px solid #000; border-bottom: 2px solid #000; }}
.listing .stat:nth-child(2n) {{ border-right: none; }}
.listing .stat:nth-last-child(-n+2) {{ border-bottom: none; }}
.listing .stat b {{ display: block; font-family: 'Anton', sans-serif; font-size: 2rem; font-weight: 400; line-height: 1.05; }}
.listing .stat small {{ color: {MUTED}; font-size: .9rem; }}
.listing .scale {{ position: relative; padding-top: 1.9rem; }}
.listing .bar {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }}
.listing .seg {{ height: 22px; border-radius: 999px; border: 2px solid #000; background: #fff; }}
.listing .seg.on {{ background: var(--v); }}
.listing .labels {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: .5rem; font-size: .9rem; color: {MUTED}; }}
.listing .labels .on {{ color: #000; font-weight: 800; }}
.listing .marker {{ position: absolute; top: 0; left: var(--pos); transform: translateX(-50%); background: #000; color: #fff;
  font-weight: 800; font-size: .8rem; border-radius: 999px; padding: .15rem .55rem; animation: as-slide 1s cubic-bezier(.2,.8,.2,1) both; }}
.listing .marker::after {{ content: ""; position: absolute; left: 50%; bottom: -6px; margin-left: -6px; border: 6px solid transparent; border-bottom: 0; border-top-color: #000; }}
@keyframes as-slide {{ from {{ left: 0; }} to {{ left: var(--pos); }} }}
@media (prefers-reduced-motion: reduce) {{ .listing .marker {{ animation: none; }} }}
.listing .caveat {{ color: {MUTED}; font-size: .9rem; margin-top: 1.2rem; max-width: 92ch; line-height: 1.5; }}
@media (max-width: 720px) {{ .listing {{ padding: 1.3rem; }} .listing .l-main {{ grid-template-columns: 1fr; gap: 1rem; }} }}

.tips {{ display: grid; gap: .9rem; }}
.tip {{ background: #fff; border: 2px solid #000; border-radius: 22px; padding: .95rem 1.15rem; line-height: 1.5; color: #000; }}
.tip b {{ display: block; font-weight: 800; font-size: 1.05rem; margin-bottom: .15rem; }}
.tip b::before {{ content: ""; display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; border: 2px solid #000;
  background: var(--c, {BLUE}); margin-right: .5rem; vertical-align: .05rem; }}
.chart-title {{ font-family: 'Archivo', sans-serif; font-weight: 800; color: #000; margin: .8rem 0 .1rem; font-size: 1.1rem; }}
.chart-note {{ font-family: 'Archivo', sans-serif; color: {MUTED}; font-size: .88rem; margin-bottom: .4rem; }}
</style>
"""


def theme():
    # autosize "contains: padding" makes Vega fit the axis labels inside the width instead of cutting them off
    return {"background": None, "autosize": {"type": "fit", "contains": "padding"},
            "config": {"font": "Archivo, Segoe UI, sans-serif", "view": {"stroke": None},
                       "axis": {"labelColor": MUTED, "titleColor": MUTED, "gridColor": "#E3EAF4", "domainColor": "#000",
                                "tickColor": "#000", "labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 500, "titlePadding": 10},
                       "legend": {"labelColor": INK, "labelFontSize": 12, "orient": "bottom", "title": None},
                       "padding": {"left": 6, "right": 12, "top": 12, "bottom": 6}}}


try:
    alt.theme.register("appscout", enable=True)(theme)
except AttributeError:  # Altair < 5.5
    alt.themes.register("appscout", theme)
    alt.themes.enable("appscout")


def title_html(title: str, note: str) -> str:
    """Chart titles live in the page, not in the chart, so they can never be clipped."""
    return f"<div class='chart-title'>{esc(title)}</div><div class='chart-note'>{esc(note)}</div>"


def listing_html(r: dict, idea: str, content: str, results: dict) -> str:
    """The idea shown the way the store would list it, with the verdict scale under it."""
    cat, stats, mk = r["category"], r["category_stats"], r["market"]
    colour = VERDICT_COLOUR[r["verdict"]]
    band = min(int(r["score"] // 25), 3)
    words = E.normalise(idea).split()
    initials = "".join(w[0] for w in words[:2]).upper()
    guess = r["category_guess"]
    reads = f"reads {guess[cat]:.0%} {esc(nice(cat))}" if cat in guess else "category chosen by you"
    segs = "".join(f"<div class='seg{' on' if i == band else ''}'></div>" for i in range(4))
    labs = "".join(f"<div class='{'on' if i == band else ''}'>{n}</div>" for i, (n, _) in enumerate(BANDS))
    title = idea.strip()[:1].upper() + idea.strip()[1:]
    val = results["prove"]["score_validation"]
    base_rate = results["data"]["kaggle"]["success_share"]
    return f"""<div class="as-root"><div class="listing" style="--v:{colour}">
<div class="l-top">
<div class="icon" aria-hidden="true">{esc(initials)}</div>
<div class="l-name"><div class="title">{esc(title)}</div><div class="cat"><span class="pill">{esc(nice(cat))}</span>{reads}</div></div>
<div class="verdict">{r['verdict']}</div>
</div>
<div class="l-main">
<div class="score"><span class="num">{r['score']:.0f}</span><span class="den">/100<br>viability</span></div>
<div class="stats">
<div class="stat"><b>{r['p_success']:.0%}</b><small>chance of reaching 1M installs (a typical app: {base_rate:.0%})</small></div>
<div class="stat"><b>{mk['neighbour_installs']:,}</b><small>installs of the 25 most similar apps</small></div>
<div class="stat"><b>{stats['success_rate']:.0%}</b><small>of {esc(nice(cat))} apps reach 1M</small></div>
<div class="stat"><b>{esc(content)}</b><small>content rating</small></div>
</div></div>
<div class="scale" style="--pos:{r['score']:.1f}%" role="img" aria-label="Verdict: {r['verdict']}, {r['score']:.0f} out of 100">
<div class="marker">{r['score']:.0f}</div><div class="bar">{segs}</div><div class="labels">{labs}</div>
</div>
<div class="caveat">The score ranks this idea among every existing app judged the same way. On apps it never saw, it told
winners from the rest with AUC {val['2018 held-out apps']['auc_score']:.2f}, and {val['2026 apps released after the snapshot']['auc_score']:.2f}
on apps launched after its data was collected. Execution, marketing and timing are invisible before launch, so read this
as a verdict, not a forecast.</div>
</div></div>"""


def how_judged_md(r: dict) -> str:
    mk = r["market"]
    return f"""
**How it was judged.** A logistic regression on what is known before launch: the category, content rating, price
and the 25 existing apps whose names are closest to your idea. Together the bars add up to the {r['p_success']:.0%} above.

- **Similar apps' installs:** about **{mk['neighbour_installs']:,}** each, more than **{mk['demand_pct']:.0f}%** of neighbourhoods. This matters most.
- **Crowding:** the niche is more crowded than **{mk['crowding_pct']:.0f}%** of niches. Crowded niches are usually ones with demand.
- **Their rating:** **{mk['neighbour_rating']:.2f}** stars on average.
"""


def drivers_chart(r: dict):
    d = r["drivers"]["contributions"]
    drv = pd.DataFrame({"input": [{"How crowded the niche is": "Niche crowding"}.get(n, n) for n in d], "push": list(d.values())})
    drv["direction"] = np.where(drv["push"] >= 0, "raises the chance", "lowers the chance")
    return alt.Chart(drv).mark_bar(cornerRadius=6, height=24, stroke="#000", strokeWidth=1.5).encode(
        x=alt.X("push:Q", title="lowers the chance of 1M installs  |  raises it"),
        y=alt.Y("input:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises the chance", "lowers the chance"],
                                                       range=["#1FB26B", "#F0442E"]), legend=None),
        tooltip=["input", alt.Tooltip("push:Q", format="+.2f")]).properties(height=260)


def tips(r: dict, paid: bool, results: dict) -> list[tuple[str, str]]:
    """Data-driven suggestions, each a (headline, explanation) pair."""
    cat, stats, sug = r["category"], r["category_stats"], r["suggestions"]
    out = []
    best = sug["niches"].iloc[0] if len(sug["niches"]) else None
    if best is not None and best["success_rate"] > stats["success_rate"] + 0.05:
        out.append(("Angle it towards " + best["genre"],
                    f"Your idea sits close to this genre, where {best['success_rate']:.0%} of apps reach a million installs "
                    f"(against {stats['success_rate']:.0%} in {nice(cat)})."))
    if sug["winning_words"]:
        out.append(("Describe it the way the winners nearby do",
                    "Successful apps close to your idea call themselves: " + ", ".join(sug["winning_words"]) + "."))
    if sug["complaints"]:
        out.append(("Solve what users complain about",
                    f"Unhappy reviews in {nice(cat)} keep coming back to: " + ", ".join(sug["complaints"][:6])
                    + ". An app that gets these right has a reason to exist."))
    if paid or (stats["success_paid"] is not None and stats["success_paid"] < stats["success_free"]):
        paid_txt = f"{stats['success_paid']:.0%}" if stats["success_paid"] is not None else "too few to measure"
        out.append(("Launch free", f"In {nice(cat)}, {stats['success_free']:.0%} of free apps reach a million installs; paid apps {paid_txt}. "
                    f"By 2026 only {results['explore']['then_now']['paid_share_2026']:.0%} of apps found by search were paid, "
                    "so earn from subscriptions or in-app purchases instead."))
    if sug["alt_categories"]:
        out.append(("Try it in another category", "It also reads as " + " or ".join(nice(c) for c in sug["alt_categories"])
                    + ". Choose one as the category to compare."))
    return out


def tips_html(items) -> str:
    dots = ["#FF6A2C", "#FFD233", "#1FB26B", "#2D6BFF", "#6C5CFF"]
    return ("<div class='as-root tips'>" + "".join(f"<div class='tip' style='--c:{dots[i % 5]}'><b>{esc(t)}</b>{esc(b)}</div>"
                                                     for i, (t, b) in enumerate(items)) + "</div>")


def niches_table(r: dict) -> pd.DataFrame:
    n = r["suggestions"]["niches"]
    return pd.DataFrame({"Genre": n["genre"], "Category": n["category"].map(nice), "Apps": n["apps"],
                         "Reach 1M": (n["success_rate"] * 100).round(0).astype(int).astype(str) + "%",
                         "Rating": n["mean_rating"].map(lambda v: f"{v:.2f}"),
                         "Closeness": (n["closeness"] * 100).round(0).astype(int).astype(str) + "%"})


def competitors_table(r: dict) -> pd.DataFrame:
    comp = r["competitors"].head(10)
    return pd.DataFrame({"App": comp["app"], "Category": comp["category"].map(nice),
                         "Installs": comp["installs"].map(lambda v: f"{v:,}+"),
                         "Rating": comp["rating"].map(lambda v: "no rating yet" if pd.isna(v) else f"{v:.1f}"),
                         "Similarity": (comp["similarity"] * 100).round(0).astype(int).astype(str) + "%"})


def live_competitors(idea: str) -> pd.DataFrame | None:
    """Today's Play Store search results for the idea. None when the store cannot be reached."""
    try:
        from google_play_scraper import search
        hits = search(idea, n_hits=10, lang="en", country="us")
    except Exception:  # ponytail: offline demo still works, the section just says so
        return None
    return pd.DataFrame([{"App": h.get("title"), "Genre": h.get("genre"), "Installs": h.get("installs"),
                          "Rating": f"{h['score']:.2f}" if h.get("score") else "no rating yet",
                          "Price": "Free" if h.get("free") else f"${h.get('price')}"} for h in hits])


def market_charts(eng, r: dict) -> list[tuple[str, str, object]]:
    """The four market charts as (title, note, chart)."""
    apps, cat, stats = eng.apps, r["category"], r["category_stats"]
    colour = VERDICT_COLOUR[r["verdict"]]
    short = {v: (f"{v // 1_000_000_000}B" if v >= 1e9 else f"{v // 1_000_000}M" if v >= 1e6 else f"{v // 1000}K" if v >= 1000 else f"{v}")
             for v in apps["installs"].unique()}
    order = [short[v] for v in sorted(short)]
    scope = apps.assign(scope=np.where(apps["category"] == cat, nice(cat), "Whole store"))
    hist = scope.groupby(["scope", "installs"]).size().rename("n").reset_index()
    hist["share"] = hist["n"] / hist.groupby("scope")["n"].transform("sum")
    hist["bucket"] = hist["installs"].map(short)
    installs = (alt.Chart(hist).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("bucket:N", sort=order, title="installs, at least", axis=alt.Axis(labelAngle=0, labelOverlap="greedy")),
        xOffset=alt.XOffset("scope:N", sort=[nice(cat), "Whole store"]),
        y=alt.Y("share:Q", axis=alt.Axis(format="%"), title="share of apps"),
        color=alt.Color("scope:N", sort=[nice(cat), "Whole store"], scale=alt.Scale(range=[BLUE, "#B9C6D8"])),
        tooltip=["scope", "bucket", alt.Tooltip("share:Q", format=".1%")])
        # grouped bars size by step, which ignores the autosize padding rule: reserve the axis room explicitly
        .properties(height=290, padding={"left": 34, "right": 12, "top": 12, "bottom": 6}))
    rat = apps[(apps["category"] == cat) & apps["rating"].notna()]
    rating = alt.layer(alt.Chart(rat).mark_bar(color=BLUE, stroke="#000", strokeWidth=1).encode(
        x=alt.X("rating:Q", bin=alt.Bin(step=0.2, extent=[1, 5]), title="rating"), y=alt.Y("count():Q", title="apps")),
        alt.Chart(pd.DataFrame({"m": [stats["mean_rating"]]})).mark_rule(color="#000", strokeWidth=3, strokeDash=[6, 4]).encode(x="m:Q")).properties(height=290)
    fp = apps[apps["category"] == cat].groupby("paid")["success"].agg(["mean", "size"]).reindex([0, 1]).fillna(0).reset_index()
    fp["model"] = fp["paid"].map({0: "Free", 1: "Paid"})
    fp["label"] = fp.apply(lambda x: f"{x['mean']:.0%} of {int(x['size'])} apps", axis=1)
    base = alt.Chart(fp).encode(y=alt.Y("model:N", title=None, sort=["Free", "Paid"]),
                                x=alt.X("mean:Q", axis=alt.Axis(format="%"), title="reached 1M installs",
                                        scale=alt.Scale(domain=[0, max(0.1, fp["mean"].max() * 1.4)])))
    freepaid = (base.mark_bar(height=36, cornerRadiusEnd=8, stroke="#000", strokeWidth=1.5).encode(
        color=alt.Color("model:N", scale=alt.Scale(domain=["Free", "Paid"], range=["#1FB26B", "#FF9F1C"]), legend=None))
        + base.mark_text(align="left", dx=8, color=INK, fontSize=13, fontWeight=600).encode(text="label:N")).properties(height=150)
    market_genres = set(apps.loc[apps["category"] == cat, "genre"])
    gt = eng.genres.assign(role=lambda x: np.where(x["genre"].isin(market_genres), "your market",
                                                   np.where(x["genre"].isin(r["suggestions"]["niches"]["genre"]), "suggested niche", "other genres")))
    genres = alt.Chart(gt).mark_circle(size=150, opacity=1, stroke="#000", strokeWidth=1.2).encode(
        x=alt.X("apps:Q", scale=alt.Scale(type="log"), title="apps in the genre (log scale)"),
        y=alt.Y("success_rate:Q", axis=alt.Axis(format="%"), title="reached 1M installs"),
        color=alt.Color("role:N", scale=alt.Scale(domain=["your market", "suggested niche", "other genres"],
                                                  range=[colour, "#1FB26B", "#E6ECF5"])),
        order=alt.Order("role:N", sort="descending"),
        tooltip=["genre", "apps", alt.Tooltip("success_rate:Q", format=".0%"), alt.Tooltip("mean_rating:Q", format=".2f")]).properties(height=290)
    return [("How far apps get", f"Share of apps in each installs bucket (at least this many): {nice(cat)} against the whole store.", installs),
            ("The rating bar to clear", f"Ratings of {nice(cat)} apps. The dashed line marks the mean, {stats['mean_rating']}.", rating),
            ("Free against paid", f"How often each business model reached a million installs in {nice(cat)}.", freepaid),
            ("Every genre: competition against demand",
             "Higher means more apps reached a million installs; further right means more apps compete.", genres)]


def data_limits_md(results: dict) -> str:
    k = results["data"]
    return f"""
* **2018 snapshot:** {k['kaggle']['apps']:,} apps and {k['reviews']['rows_with_text']:,} reviews scraped from the Google Play Store
  (Kaggle, Lavanya Gupta, CC BY 3.0). Only apps that existed then are in it; apps already removed are not.
* **Reviews** cover {k['reviews']['apps_with_text']} apps, mostly popular ones, so complaint topics describe established competitors.
* **2026 scrape:** {k['scrape']['apps_with_details']:,} apps from {k['scrape']['queries']} searches on {k['scrape']['scraped_at'][:10]}.
  Search favours popular apps, so it is used for comparison and testing, not for totals.
* Every number here comes from the project notebooks via `results/results.json`.
"""
