"""AppScout: type a Play Store app idea, get a data-backed viability verdict.

    streamlit run app/app.py

What is shown for an idea lives in views.py, shared with the Colab notebook; this file is the
Streamlit page around it: the landing hero, the form, the tabs and the footer.
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT / "app")]
import common as C  # noqa: E402
import engine as E  # noqa: E402
import views as V  # noqa: E402

REPO = "https://github.com/Johnchrispindurai/appscout"
COLAB = "https://colab.research.google.com/github/Johnchrispindurai/appscout/blob/main/AppScout_Colab.ipynb"
TEAM = ["John Chrispin Durai A", "Leonal Paul J", "Jebarson P"]

st.set_page_config(page_title="AppScout", page_icon="📱", layout="wide")
RESULTS = json.loads(C.RESULTS.read_text(encoding="utf-8"))
k, val = RESULTS["data"], RESULTS["prove"]["score_validation"]
nice = V.nice


def html(s: str) -> None:
    """Raw HTML, flattened: Markdown would turn indented or blank-line-separated HTML into code blocks."""
    st.markdown("\n".join(line.strip() for line in s.splitlines() if line.strip()), unsafe_allow_html=True)


def chart(title, note, c):
    html(V.title_html(title, note))
    st.altair_chart(c, width="stretch", theme=None)


# ------------------------------------------------------------------ page style
html(V.CSS + f"""
<style>
.stApp {{ background: #fff; }}
[data-testid="stHeader"] {{ display: none; }}
.block-container {{ max-width: 1360px; padding: .8rem 1.2rem 3rem; }}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp [data-baseweb="select"] div,
.stApp [data-testid="stMarkdownContainer"] {{ font-family: 'Archivo', 'Segoe UI', sans-serif; }}
a {{ color: inherit; }}

.ticker {{ background: {V.LILAC}; border: 2px solid #000; border-radius: 999px; overflow: hidden; white-space: nowrap; }}
.ticker .track {{ display: inline-flex; gap: 2.4rem; padding: .45rem 0; animation: ticker 38s linear infinite; }}
.ticker span {{ font-weight: 800; font-size: .82rem; letter-spacing: .04em; text-transform: uppercase; }}
.ticker span::before {{ content: "▲"; margin-right: .7rem; color: {V.BLUE}; font-size: .7rem; }}
@keyframes ticker {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}

.hero {{ position: relative; background: {V.SKY}; border-radius: 40px; margin-top: .7rem; overflow: hidden; min-height: 640px; }}
.hero .tube {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.nav {{ position: relative; z-index: 3; display: flex; justify-content: space-between; align-items: center; padding: 1.3rem 1.6rem; gap: 1rem; }}
.logo {{ width: 52px; height: 52px; border-radius: 50%; border: 2.5px solid #000; background: #fff; display: grid; place-items: center; }}
.links {{ display: flex; gap: .5rem; flex-wrap: wrap; justify-content: flex-end; }}
.pill {{ display: inline-block; border: 2.5px solid #000; border-radius: 999px; background: #fff; color: #000 !important;
  padding: .55rem 1.25rem; font-weight: 800; font-size: .88rem; text-decoration: none !important; letter-spacing: .02em; text-transform: uppercase; }}
.pill:hover, .pill:focus-visible {{ background: #000; color: #fff !important; outline: none; }}
.pill.dark {{ background: #000; color: #fff !important; }}
.pill.dark:hover {{ background: {V.BLUE}; }}
.pill.big {{ padding: .95rem 1.9rem; font-size: 1rem; }}
.stage {{ position: relative; z-index: 2; text-align: center; padding: 3.2rem 1rem 4.2rem; }}
.mega {{ font-family: 'Anton', sans-serif !important; font-weight: 400 !important; font-size: clamp(5rem, 17vw, 16.5rem) !important;
  line-height: .86 !important; letter-spacing: -0.005em; color: #000 !important; margin: 0 !important; padding: 0 !important; }}
.stage p.tag {{ font-family: 'Archivo', sans-serif; font-weight: 600; font-size: clamp(1.6rem, 3.8vw, 3.3rem) !important; line-height: 1.1 !important;
  letter-spacing: -0.025em; margin: 1.3rem 0 2rem !important; color: #000; }}
.ctas {{ display: flex; gap: .8rem; justify-content: center; flex-wrap: wrap; }}
.sticker {{ position: absolute; z-index: 3; width: clamp(64px, 9vw, 128px); animation: bob 5.5s ease-in-out infinite alternate; }}
.sticker svg {{ width: 100%; height: auto; display: block; }}
.s1 {{ top: 4%; left: 6%; --r: -14deg; }} .s2 {{ top: 2%; right: 7%; --r: 12deg; animation-delay: -1.5s; }}
.s3 {{ bottom: 12%; left: 9%; --r: 9deg; animation-delay: -3s; }} .s4 {{ bottom: 15%; right: 9%; --r: -10deg; animation-delay: -4.2s; }}
@keyframes bob {{ from {{ transform: rotate(var(--r)) translateY(0); }} to {{ transform: rotate(var(--r)) translateY(-12px); }} }}
@media (prefers-reduced-motion: reduce) {{ .ticker .track, .sticker {{ animation: none; }} .sticker {{ transform: rotate(var(--r)); }} }}
@media (max-width: 760px) {{ .links .pill:not(.dark) {{ display: none; }} .hero {{ min-height: 560px; }} .s3, .s4 {{ display: none; }} }}

.sec-h {{ font-family: 'Anton', sans-serif !important; font-weight: 400 !important; font-size: clamp(2.6rem, 6vw, 5rem) !important;
  line-height: .92 !important; color: #000 !important; text-transform: uppercase; margin: 3.4rem 0 .6rem !important; padding: 0 !important; }}
.sec-sub {{ color: {V.MUTED}; font-size: 1.1rem; max-width: 60ch; margin-bottom: 1.4rem; }}
.steps {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }}
.step {{ border: 2.5px solid #000; border-radius: 30px; padding: 1.5rem 1.5rem 1.6rem; background: var(--bg); }}
.step .n {{ width: 56px; height: 56px; border-radius: 50%; border: 2.5px solid #000; background: #fff; display: grid; place-items: center;
  font-family: 'Anton', sans-serif; font-size: 1.7rem; box-shadow: 3px 3px 0 #000; }}
.step h3 {{ font-family: 'Archivo', sans-serif !important; font-weight: 800 !important; font-size: 1.45rem !important; margin: 1rem 0 .4rem !important; padding: 0 !important; color: #000 !important; }}
.step p {{ margin: 0; line-height: 1.55; color: #000; }}
.step .fig {{ display: block; margin-top: 1rem; font-family: 'Anton', sans-serif; font-size: 2.4rem; line-height: 1; }}
.step .fig small {{ font-family: 'Archivo', sans-serif; font-size: .9rem; color: {V.MUTED}; display: block; margin-top: .25rem; }}
@media (max-width: 860px) {{ .steps {{ grid-template-columns: 1fr; }} }}

[data-testid="stForm"] {{ background: #fff; border: 2.5px solid #000 !important; border-radius: 32px; padding: 1.6rem 1.8rem 1.1rem; box-shadow: 8px 8px 0 #000; }}
[data-testid="stForm"] textarea {{ font-size: 1.3rem !important; font-weight: 500; }}
[data-baseweb="textarea"], [data-baseweb="select"] > div, [data-baseweb="input"] {{ border: 2px solid #000 !important; border-radius: 18px !important; background: #F3F8FF !important; }}
[data-testid="stWidgetLabel"] p {{ font-weight: 700 !important; }}
[data-testid="stNumberInputContainer"] {{ border: 2px solid #000; border-radius: 18px; overflow: hidden; background: #F3F8FF; }}
[data-testid="stNumberInputContainer"] [data-baseweb="input"] {{ border: none !important; border-radius: 0 !important; }}
[data-testid="stFormSubmitButton"] button {{ background: #000 !important; color: #fff !important; border: 2.5px solid #000 !important; border-radius: 999px !important;
  padding: .75rem 2.1rem !important; }}
[data-testid="stFormSubmitButton"] button p {{ font-weight: 800 !important; font-size: 1.02rem !important; letter-spacing: .03em; text-transform: uppercase; }}
[data-testid="stFormSubmitButton"] button:hover {{ background: {V.BLUE} !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .55rem; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ border: 2.5px solid #000 !important; border-radius: 999px; padding: .5rem 1.25rem !important; background: #fff; height: auto; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 800 !important; font-size: .98rem !important; }}
.stTabs [aria-selected="true"] {{ background: #000 !important; }}
.stTabs [aria-selected="true"] p {{ color: #fff !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
[data-testid="stDataFrame"] {{ border: 2px solid #000; border-radius: 18px; overflow: hidden; }}
[data-testid="stExpander"] details {{ border: 2px solid #000 !important; border-radius: 22px !important; }}
[data-testid="stAlert"] {{ border: 2px solid #000; border-radius: 18px; }}

.foot {{ background: #000; color: #fff; border-radius: 40px; padding: 2.6rem 2.4rem 2rem; margin-top: 3rem; }}
.foot .big {{ font-family: 'Anton', sans-serif; font-size: clamp(3rem, 8.5vw, 7.5rem); line-height: .9; text-transform: uppercase; }}
.foot .cols {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.6rem; margin-top: 2.2rem; border-top: 2px solid #fff; padding-top: 1.4rem; }}
.foot h4 {{ font-family: 'Archivo', sans-serif !important; color: #fff !important; font-weight: 800 !important; font-size: 1rem !important; margin: 0 0 .5rem !important; padding: 0 !important; }}
.foot p, .foot a {{ color: #C9D2E3 !important; margin: 0 0 .2rem; line-height: 1.5; }}
.foot a.pill {{ margin: .6rem .4rem 0 0; color: #000 !important; background: #fff; }}
.foot a.pill:hover, .foot a.pill:focus-visible {{ background: {V.BLUE}; color: #fff !important; }}
@media (max-width: 760px) {{ .foot .cols {{ grid-template-columns: 1fr; }} }}
</style>""")

# ------------------------------------------------------------------ hero
TUBE = """<svg class="tube" viewBox="0 0 1400 720" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
<defs>
<filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="14"/></filter>
<filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="7"/></filter>
<filter id="grain" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency="1.1" numOctaves="2" seed="7"/>
<feColorMatrix values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 1.6 -0.75"/><feComposite in2="SourceGraphic" operator="in"/></filter>
<path id="p" d="M -160 610 C 120 640, 230 330, 430 190 C 610 60, 860 50, 920 250 C 975 440, 1120 600, 1290 420 C 1380 320, 1420 190, 1580 170"/>
</defs>
<use href="#p" fill="none" stroke="#0F2E8C" stroke-opacity=".35" stroke-width="150" stroke-linecap="round" filter="url(#soft)" transform="translate(0 28)"/>
<use href="#p" fill="none" stroke="#1D4FE0" stroke-width="146" stroke-linecap="round"/>
<use href="#p" fill="none" stroke="#3B7DFF" stroke-width="112" stroke-linecap="round" transform="translate(-4 -12)"/>
<use href="#p" fill="none" stroke="#9CC3FF" stroke-opacity=".85" stroke-width="24" stroke-linecap="round" filter="url(#glow)" transform="translate(-10 -44)"/>
<use href="#p" fill="none" stroke="#fff" stroke-width="146" stroke-linecap="round" filter="url(#grain)" opacity=".22"/>
</svg>"""
STICKERS = [
    ("s1", """<svg viewBox="0 0 124 124"><rect x="14" y="16" width="100" height="100" rx="28" fill="#B9461C" stroke="#000" stroke-width="4"/>
<rect x="6" y="6" width="100" height="100" rx="28" fill="#FF6A2C" stroke="#000" stroke-width="4"/>
<g fill="#fff" stroke="#000" stroke-width="3">
<rect x="22" y="22" width="18" height="18" rx="6"/><rect x="47" y="22" width="18" height="18" rx="6"/><rect x="72" y="22" width="18" height="18" rx="6"/>
<rect x="22" y="47" width="18" height="18" rx="6"/><rect x="47" y="47" width="18" height="18" rx="6" fill="#FFD233"/><rect x="72" y="47" width="18" height="18" rx="6"/>
<rect x="22" y="72" width="18" height="18" rx="6"/><rect x="47" y="72" width="18" height="18" rx="6"/><rect x="72" y="72" width="18" height="18" rx="6"/></g></svg>"""),
    ("s2", """<svg viewBox="0 0 124 124"><circle cx="66" cy="68" r="50" fill="#C99500" stroke="#000" stroke-width="4"/>
<circle cx="58" cy="58" r="50" fill="#FFD233" stroke="#000" stroke-width="4"/><circle cx="58" cy="58" r="36" fill="none" stroke="#000" stroke-width="3" stroke-dasharray="5 6"/>
<path d="M58 30 L66.5 49 L87 51 L71.5 64.5 L76 85 L58 74.5 L40 85 L44.5 64.5 L29 51 L49.5 49 Z" fill="#000"/></svg>"""),
    ("s3", """<svg viewBox="0 0 140 110"><rect x="14" y="16" width="120" height="86" rx="30" fill="#137A48" stroke="#000" stroke-width="4"/>
<rect x="6" y="6" width="120" height="86" rx="30" fill="#1FB26B" stroke="#000" stroke-width="4"/>
<path d="M36 28 v26 m-12 -11 l12 13 l12 -13" fill="none" stroke="#fff" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
<text x="58" y="66" font-family="Anton, sans-serif" font-size="36" fill="#fff" stroke="#000" stroke-width="1.2">1M+</text></svg>"""),
    ("s4", """<svg viewBox="0 0 130 130"><path d="M84 84 L118 118" stroke="#000" stroke-width="16" stroke-linecap="round"/>
<circle cx="58" cy="60" r="44" fill="#4B3CCB" stroke="#000" stroke-width="4"/><circle cx="52" cy="52" r="44" fill="#6C5CFF" stroke="#000" stroke-width="4"/>
<circle cx="52" cy="52" r="27" fill="#D8E9FF" stroke="#000" stroke-width="4"/><path d="M38 44 a16 16 0 0 1 14 -10" fill="none" stroke="#fff" stroke-width="6" stroke-linecap="round"/></svg>"""),
]
LOGO = """<svg width="28" height="28" viewBox="0 0 28 28"><rect x="3" y="3" width="22" height="22" rx="7" fill="#2D6BFF" stroke="#000" stroke-width="2.5"/>
<circle cx="13" cy="13" r="5" fill="#fff" stroke="#000" stroke-width="2"/><path d="M17 17 L21 21" stroke="#000" stroke-width="2.5" stroke-linecap="round"/></svg>"""
facts = [f"{k['kaggle']['apps']:,} apps scored", f"{k['reviews']['rows_with_text']:,} reviews read", "Live Play Store search",
         f"AUC {val['2018 held-out apps']['auc_score']:.2f} on unseen apps", f"{k['scrape']['apps_with_details']:,} apps scraped in 2026",
         "33 categories", "Data science ecosystem project"]
ticker = "".join(f"<span>{f}</span>" for f in facts * 2)
html(f"""<div id="top" class="ticker" aria-hidden="true"><div class="track">{ticker}</div></div>
<section class="hero">{TUBE}
<nav class="nav" aria-label="Main"><a class="logo" href="#top" aria-label="AppScout home">{LOGO}</a>
<div class="links"><a class="pill" href="#how">How it works</a><a class="pill" href="#check">Check an idea</a>
<a class="pill" href="{COLAB}" target="_blank" rel="noopener">Colab</a><a class="pill" href="{REPO}" target="_blank" rel="noopener">GitHub</a>
<a class="pill dark" href="#check">Launch check</a></div></nav>
<div class="stage">{''.join(f'<div class="sticker {c}" aria-hidden="true">{s}</div>' for c, s in STICKERS)}
<h1 class="mega">APPSCOUT</h1>
<p class="tag">Your app idea. Tested.</p>
<div class="ctas"><a class="pill big dark" href="#check">Check my idea</a><a class="pill big" href="{COLAB}" target="_blank" rel="noopener">Run it in Colab</a></div>
</div></section>""")

# ------------------------------------------------------------------ how it works (a real sequence, so numbered)
ct = RESULTS["classify"]["category_test"]
html(f"""<h2 id="how" class="sec-h">Three steps. No guesswork.</h2>
<p class="sec-sub">AppScout judges an idea the way the Play Store already judged the apps before it.</p>
<div class="steps">
<div class="step" style="--bg:#FFE7DA"><div class="n">1</div><h3>Describe it</h3>
<p>Type the app in one sentence. AppScout reads it the way the store reads app names and places it in one of 33 categories.</p>
<span class="fig">{ct['balanced_top3']:.0%}<small>right within its top three guesses</small></span></div>
<div class="step" style="--bg:#FFF4C7"><div class="n">2</div><h3>Meet its neighbours</h3>
<p>The 25 existing apps with the closest names become its neighbourhood: what they reached, how they are rated, how crowded it is.</p>
<span class="fig">{k['kaggle']['apps']:,}<small>apps and {k['reviews']['rows_with_text']:,} reviews to compare with</small></span></div>
<div class="step" style="--bg:#DDF5E8"><div class="n">3</div><h3>Get the verdict</h3>
<p>A model trained only on what is knowable before launch gives the chance of a million installs, the reasons, and better directions.</p>
<span class="fig">AUC {val['2018 held-out apps']['auc_score']:.2f}<small>on apps it never saw ({val['2026 apps released after the snapshot']['auc_score']:.2f} on apps launched after its data)</small></span></div>
</div>
<h2 id="check" class="sec-h">What are you building?</h2>
<p class="sec-sub">One sentence is enough. Use the words someone would type into the store's search box.</p>""")


# ------------------------------------------------------------------ the check
@st.cache_resource(show_spinner="Reading 9,659 apps and 37,427 reviews (about 20 seconds, once)...")
def engine():
    return E.fit(C.load_apps(), C.load_reviews())


live_competitors = st.cache_data(ttl=3600, show_spinner=False)(V.live_competitors)
eng = engine()

with st.form("idea"):
    idea = st.text_area("Your idea, in one sentence", "recipe app that plans meals from what is in your fridge", height=80)
    c1, c2, c3, c4 = st.columns(4)
    cats = sorted(eng.apps["category"].unique())
    cat_choice = c1.selectbox("Category", ["Detect from my idea"] + cats, format_func=lambda c: c if c.startswith("Detect") else nice(c))
    model = c2.radio("Business model", ["Free", "Paid"], horizontal=True)
    price = c3.number_input("Price (USD)", 0.0, 50.0, 2.99, 0.5, disabled=model == "Free")
    content = c4.selectbox("Content rating", V.CONTENT)
    live = st.checkbox("Also search the Play Store for today's competitors", value=True)
    st.form_submit_button("Check my idea", type="primary")

if not E.normalise(idea):
    st.warning("Describe what the app does, for example \"meal planner for students\". "
               "Words like app, free or best alone say nothing about the market.")
    st.stop()

r = eng.evaluate(idea, category=None if cat_choice.startswith("Detect") else cat_choice, paid=model == "Paid",
                 price=price if model == "Paid" else 0.0, content=content)
html(V.listing_html(r, idea, content, RESULTS))
if r["match_quality"] < 0.25:
    st.warning("Few existing apps have names like this idea, so the comparison is weak. "
               "Try the words a user would type into the store's search box.")

why, better, rivals, market = st.tabs(["Why this score", "Make it better", "Competitors", f"The {nice(r['category'])} market"])

with why:
    a, b = st.columns([1.35, 1], gap="large")
    with a:
        chart("What moved the score", "Each bar is one input's exact share of the answer, measured against an average app (log-odds).",
              V.drivers_chart(r))
    b.markdown(V.how_judged_md(r))

with better:
    a, b = st.columns([1, 1], gap="large")
    with a:
        html(V.tips_html(V.tips(r, model == "Paid", RESULTS)))
    with b:
        html(V.title_html("Better-aligned niches", "Genres close to your idea, ranked by closeness times how often "
                          "apps there reach a million installs."))
        st.dataframe(V.niches_table(r), hide_index=True, width="stretch")
        niches = r["suggestions"]["niches"]
        segment = RESULTS["cluster"]["genre_segment"].get(niches["genre"].iloc[0]) if len(niches) else None
        if segment:
            st.caption(f"{niches['genre'].iloc[0]} belongs to the market segment \"{segment}\" found by clustering.")

with rivals:
    html(V.title_html("Closest apps in the 2018 data", "The 10 names most similar to your idea, with what they achieved."))
    st.dataframe(V.competitors_table(r), hide_index=True, width="stretch")
    html(V.title_html("On the Play Store today", "Top search results for your idea's words, fetched live."))
    now = live_competitors(idea) if live else None
    if not live:
        st.info("Live search is off. Tick the box above the button to include it.")
    elif now is None:
        st.info("The Play Store could not be reached, so live results are left out. The 2018 comparison is unaffected.")
    else:
        st.dataframe(now, hide_index=True, width="stretch")

with market:
    charts = V.market_charts(eng, r)
    for row in (charts[:2], charts[2:]):
        for col, (title, note, c) in zip(st.columns(2, gap="large"), row):
            with col:
                chart(title, note, c)

st.write("")
with st.expander("About the data and its limits"):
    st.markdown(V.data_limits_md(RESULTS))

# ------------------------------------------------------------------ footer
html(f"""<footer class="foot"><div class="big">Know before you build.</div>
<div class="cols">
<div><h4>Built by</h4>{''.join(f'<p>{n}</p>' for n in TEAM)}</div>
<div><h4>Course project</h4><p>Data Science Ecosystem (23CS2024)</p><p>Karunya Institute of Technology and Sciences</p><p>October 2026</p></div>
<div><h4>Data</h4><p>Google Play Store apps and reviews, 2018 (Kaggle, CC BY 3.0)</p><p>{k['scrape']['apps_with_details']:,} apps scraped from the store in 2026</p>
<a class="pill" href="{REPO}" target="_blank" rel="noopener">GitHub</a><a class="pill" href="{COLAB}" target="_blank" rel="noopener">Colab</a></div>
</div></footer>""")
