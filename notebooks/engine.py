"""The viability engine: an app idea in, a scored and explained verdict out.

Used by the notebooks (which measure how well it works) and by the web app (which shows
it), so both always give the same numbers.

    eng = fit(apps, reviews)
    eng.evaluate("pomodoro study timer for students")

How an idea is judged, all from the 2018 Play Store snapshot:
1. The idea text is turned into a TF-IDF vector of character n-grams, the same way as
   every app name in the store.
2. A logistic regression on app names guesses the category (the user can override it).
3. The K most similar app names are the idea's neighbourhood, its likely competitors.
   Their installs, ratings and how tightly they crowd around the idea are three
   neighbourhood signals.
4. A logistic regression on what is known before launch (category, content rating,
   price and the neighbourhood signals) gives the probability of reaching 1M installs.
   The viability score is that probability's percentile among existing apps, and its
   breakdown is the model's exact contribution from each input.
5. The neighbourhood signals are also shown on their own, as market context, and drive
   the suggestions: better-aligned niches, complaint topics, words that winners use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

K = 25                       # neighbourhood size, chosen in notebook 03
CATEGORY_C = 20.0            # regularisation of the category model, chosen in notebook 04
# Words that say nothing about what an app does.
NOISE = set(ENGLISH_STOP_WORDS) | {"app", "apps", "android", "free", "pro", "lite", "best", "new", "mobile",
                                   "application", "helps", "help", "users", "user", "people", "want", "make",
                                   "lets", "let", "using", "based", "simple", "easy", "idea", "platform"}
NUMERIC = ["nb_demand", "nb_crowding", "nb_rating", "log_price", "paid"]
# The success model's inputs, grouped the way the app explains them.
DRIVERS = {"category": "Category", "content": "Content rating", "paid": "Price model", "log_price": "Price model",
           "nb_demand": "Neighbours' installs", "nb_crowding": "How crowded the niche is", "nb_rating": "Neighbours' ratings"}


def normalise(text: str) -> str:
    """Lower-case words of letters only, noise words removed."""
    return " ".join(w for w in re.findall(r"[a-z]+", str(text).lower()) if w not in NOISE)


def pct(values, reference_sorted: np.ndarray) -> np.ndarray:
    """Percentile (0..1) of each value within a sorted reference sample, ties counted half."""
    lo = np.searchsorted(reference_sorted, values, side="left")
    hi = np.searchsorted(reference_sorted, values, side="right")
    return (lo + hi) / 2 / len(reference_sorted)


def vectorizer() -> TfidfVectorizer:
    # Character 3-5-grams within words: "calorie" and "calories" share most of their grams,
    # which matters for names of two or three words.
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)


def complaint_words(reviews: pd.DataFrame, apps: pd.DataFrame, min_neg: int = 40, top: int = 8) -> dict:
    """Per category, the words most typical of its negative reviews compared with negative
    reviews everywhere else (smoothed log-odds). Comparing negative with negative cancels the
    words that only say "this is bad" and keeps what is bad. Categories with fewer than
    `min_neg` negative reviews are left out."""
    r = reviews.merge(apps[["app", "category"]], on="app")
    cv = CountVectorizer(stop_words=list(NOISE), min_df=3, token_pattern=r"(?u)\b[a-z]{3,}\b")
    m = cv.fit_transform(r["review"].str.lower())
    words = np.array(cv.get_feature_names_out())
    negative = (r["sentiment"] == "Negative").to_numpy()
    all_neg = np.asarray(m[negative].sum(0)).ravel()
    out = {}
    for cat, g in r.groupby("category"):
        mine = negative & (r["category"] == cat).to_numpy()
        if mine.sum() < min_neg:
            continue
        n = np.asarray(m[mine].sum(0)).ravel() + 1
        o = all_neg - n + 2
        score = np.log(n / n.sum()) - np.log(o / o.sum())
        spread = sum((np.asarray(m[(mine & (r["app"] == a).to_numpy())].sum(0)).ravel() > 0) for a in g["app"].unique())
        score[(n < 6) | (spread < 3)] = -np.inf                  # a few complaints, about at least 3 apps
        out[cat] = [str(words[i]) for i in np.argsort(-score)[:top] if np.isfinite(score[i])]
    return out


@dataclass
class Engine:
    apps: pd.DataFrame
    vec: TfidfVectorizer
    X: object
    nn: NearestNeighbors
    category_model: LogisticRegression
    success_model: object
    reference: dict            # sorted leave-one-out neighbourhood signals and success probabilities of every app
    mean_row: np.ndarray       # the average training app, after the success model's preprocessing
    genres: pd.DataFrame
    complaints: dict
    k: int

    # ---------------------------------------------------------- neighbourhoods

    def neighbours(self, Xq, exclude=None, k=None):
        """(similarities, indices) of the k most similar apps per query row, skipping `exclude[i]`."""
        k = k or self.k
        dist, idx = self.nn.kneighbors(Xq, n_neighbors=k + 1)
        sim = 1 - dist
        if exclude is None:
            return sim[:, :k], idx[:, :k]
        keep = idx != np.asarray(exclude)[:, None]
        keep[keep.sum(1) > k, -1] = False                         # self not found: drop the farthest
        return sim[keep].reshape(-1, k), idx[keep].reshape(-1, k)

    def signals(self, sim, idx) -> pd.DataFrame:
        w = sim + 1e-9
        inst = self.apps["log_installs"].to_numpy()[idx]
        rat = self.apps["rating"].to_numpy()[idx]
        ok = ~np.isnan(rat)
        nb_rating = np.where(ok.any(1), np.nansum(np.where(ok, rat * w, 0), 1) / np.maximum((w * ok).sum(1), 1e-9),
                             np.nanmean(self.apps["rating"]))
        return pd.DataFrame({"nb_demand": (inst * w).sum(1) / w.sum(1), "nb_crowding": sim.mean(1),
                             "nb_rating": nb_rating})

    def rows(self, frame: pd.DataFrame, exclude=None) -> pd.DataFrame:
        """The success model's input for each app in `frame` (columns app, category, content, paid, price)."""
        Xq = self.vec.transform(frame["app"].map(normalise))
        sim, idx = self.neighbours(Xq, exclude=exclude)
        return self.signals(sim, idx).assign(category=frame["category"].to_numpy(), content=frame["content"].to_numpy(),
                                             paid=frame["paid"].to_numpy(), log_price=np.log1p(frame["price"].to_numpy()))

    # ---------------------------------------------------------- scoring

    def evaluate(self, idea: str, category: str | None = None, paid: bool = False, price: float = 0.0,
                 content: str = "Everyone") -> dict:
        xq = self.vec.transform([normalise(idea)])
        probs = pd.Series(self.category_model.predict_proba(xq)[0], self.category_model.classes_).sort_values(ascending=False)
        category = category or probs.index[0]
        sim, idx = self.neighbours(xq)
        sig = self.signals(sim, idx)
        row = sig.assign(category=category, content=content, paid=int(paid), log_price=np.log1p(price if paid else 0))
        p_success = float(self.success_model.predict_proba(row)[0, 1])
        score = float(pct(p_success, self.reference["p_success"]) * 100)
        comp = self.apps.iloc[idx[0]].assign(similarity=sim[0])
        return {
            "idea": idea, "category": category, "category_guess": probs.head(3).round(3).to_dict(),
            "match_quality": float(sim[0].max()),
            "score": round(score, 1), "verdict": verdict(score), "p_success": round(p_success, 3),
            "drivers": self.drivers(row),
            "market": self.market(sig),
            "competitors": comp[["app", "category", "genre", "installs", "rating", "reviews", "paid", "similarity"]],
            "category_stats": self.category_stats(category),
            "suggestions": self.suggest(idea, xq, category, comp, probs),
        }

    def score_many(self, frame: pd.DataFrame, exclude=None) -> pd.Series:
        """Viability scores for many apps at once, each app's own name as the idea (used to validate)."""
        p = self.success_model.predict_proba(self.rows(frame, exclude))[:, 1]
        return pd.Series(pct(p, self.reference["p_success"]) * 100, index=frame.index)

    def drivers(self, row: pd.DataFrame) -> dict:
        """The success model's log-odds for this idea, split exactly by input: each value is how far
        that input moves the idea away from an average app. Baseline + sum of drivers = log-odds."""
        pre, lr = self.success_model[0], self.success_model[-1]
        z = pre.transform(row)[0]
        contrib = (z - self.mean_row) * lr.coef_[0]
        out = {}
        for name, c in zip(pre.get_feature_names_out(), contrib):
            col = name.split("__", 1)[1]
            key = DRIVERS.get(col) or DRIVERS[col.split("_")[0] if col.startswith(("category_", "content_")) else col]
            out[key] = out.get(key, 0.0) + float(c)
        baseline = float(lr.intercept_[0] + self.mean_row @ lr.coef_[0])
        return {"baseline_log_odds": round(baseline, 4), "contributions": {k: round(v, 4) for k, v in out.items()}}

    def market(self, sig: pd.DataFrame) -> dict:
        """Neighbourhood signals as percentiles among every existing app's neighbourhood (context, not score)."""
        ref = self.reference
        return {"demand_pct": round(float(pct(sig["nb_demand"], ref["nb_demand"])[0]) * 100, 1),
                "crowding_pct": round(float(pct(sig["nb_crowding"], ref["nb_crowding"])[0]) * 100, 1),
                "neighbour_rating_pct": round(float(pct(sig["nb_rating"], ref["nb_rating"])[0]) * 100, 1),
                "neighbour_installs": int(round(10 ** float(sig["nb_demand"].iloc[0]), -2)),
                "neighbour_rating": round(float(sig["nb_rating"].iloc[0]), 2)}

    # ---------------------------------------------------------- context and suggestions

    def category_stats(self, category: str) -> dict:
        g = self.apps[self.apps["category"] == category]
        top = g["installs"].sort_values(ascending=False)
        return {"apps": int(len(g)), "median_installs": int(g["installs"].median()),
                "success_rate": round(float(g["success"].mean()), 3), "mean_rating": round(float(g["rating"].mean()), 2),
                "paid_share": round(float(g["paid"].mean()), 3),
                "top10_install_share": round(float(top.head(10).sum() / max(top.sum(), 1)), 3),
                "success_free": round(float(g.loc[g["paid"] == 0, "success"].mean()), 3),
                "success_paid": None if (g["paid"] == 1).sum() < 10 else round(float(g.loc[g["paid"] == 1, "success"].mean()), 3)}

    def suggest(self, idea: str, xq, category: str, comp: pd.DataFrame, probs: pd.Series) -> dict:
        # Better-aligned niches (a hybrid recommender): how much of the idea's wider neighbourhood
        # (4K apps) sits in each genre, times that genre's success rate (the demand signal, which
        # notebook 06 validated; crowding and rating did not predict success, so they are context only).
        sim, idx = self.neighbours(xq, k=4 * self.k)
        near = pd.Series(sim[0], self.apps["genre"].to_numpy()[idx[0]]).groupby(level=0).sum()
        g = self.genres.assign(closeness=lambda d: d["genre"].map(near / near.sum()).fillna(0))
        g["fit"] = g["closeness"] * g["success_rate"]
        niches = g[g["closeness"] >= 0.05].sort_values("fit", ascending=False).head(5)
        # What the successful neighbours call themselves that the idea does not.
        hits, miss = comp[comp["installs"] >= 1_000_000], comp[comp["installs"] < 1_000_000]
        idea_words = set(normalise(idea).split())
        words = pd.Series(" ".join(hits["app"].map(normalise)).split()).value_counts()
        lose = pd.Series(" ".join(miss["app"].map(normalise)).split()).value_counts()
        angle = [w for w in words.index if len(w) > 2 and words[w] >= 2 and words[w] > lose.get(w, 0) and w not in idea_words][:6]
        return {"niches": niches[["genre", "category", "apps", "median_installs", "success_rate", "mean_rating",
                                  "closeness"]].reset_index(drop=True),
                "complaints": self.complaints.get(category, []),
                "winning_words": angle,
                "successful_neighbours": hits["app"].head(5).tolist(),
                "alt_categories": [c for c in probs.index[:3] if c != category]}


def verdict(score: float) -> str:
    return ("Strong opportunity" if score >= 75 else "Promising" if score >= 50
            else "Tough market" if score >= 25 else "High risk")


def genre_table(apps: pd.DataFrame, min_apps: int = 20) -> pd.DataFrame:
    """Market profile of every genre with at least `min_apps` apps."""
    g = apps.groupby("genre").agg(category=("category", lambda s: s.mode()[0]), apps=("app", "size"),
                                  median_installs=("installs", "median"), success_rate=("success", "mean"),
                                  mean_rating=("rating", "mean"), paid_share=("paid", "mean"))
    return g[g["apps"] >= min_apps].reset_index()


def success_pipeline():
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["category", "content"]),
                             ("num", StandardScaler(), NUMERIC)], sparse_threshold=0)
    return make_pipeline(pre, LogisticRegression(max_iter=2000))


def fit(apps: pd.DataFrame, reviews: pd.DataFrame | None = None, k: int = K) -> Engine:
    apps = apps.reset_index(drop=True)
    vec = vectorizer()
    X = vec.fit_transform(apps["app"].map(normalise))
    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
    # Balanced class weights: FAMILY is a catch-all (a fifth of the store) and would otherwise absorb ideas.
    category_model = LogisticRegression(C=CATEGORY_C, max_iter=3000, class_weight="balanced").fit(X, apps["category"])
    eng = Engine(apps, vec, X, nn, category_model, None, {}, None, genre_table(apps), {}, k)
    # Leave-one-out: every app is described by its neighbours, never by itself.
    train = eng.rows(apps, exclude=np.arange(len(apps)))
    eng.success_model = success_pipeline().fit(train, apps["success"])
    eng.reference = {c: np.sort(train[c].to_numpy()) for c in ["nb_demand", "nb_crowding", "nb_rating"]}
    eng.reference["p_success"] = np.sort(eng.success_model.predict_proba(train)[:, 1])
    eng.mean_row = np.asarray(eng.success_model[0].transform(train)).mean(0)
    eng.complaints = complaint_words(reviews, apps) if reviews is not None else {}
    return eng


def demo() -> None:
    import common as C
    assert normalise("The BEST Calorie-Counter app 2018!") == "calorie counter"
    assert list(pct([1, 2, 5], np.array([1.0, 2, 3, 4]))) == [0.125, 0.375, 1.0]
    apps = C.load_apps()
    eng = fit(apps, C.load_reviews())
    r = eng.evaluate("calorie counter and diet tracker")
    assert r["category"] == "HEALTH_AND_FITNESS", r["category_guess"]
    assert 0 <= r["score"] <= 100 and r["verdict"] == verdict(r["score"])
    assert r["competitors"]["app"].str.lower().str.contains("calor").any()
    assert r["category_stats"]["apps"] == (apps["category"] == "HEALTH_AND_FITNESS").sum()
    # The drivers add up exactly to the model's log-odds.
    d = r["drivers"]
    logodds = d["baseline_log_odds"] + sum(d["contributions"].values())
    assert abs(1 / (1 + np.exp(-logodds)) - r["p_success"]) < 1e-3, (logodds, r["p_success"])
    # Leave-one-out really leaves the app out.
    sim, idx = eng.neighbours(eng.X[:50], exclude=np.arange(50))
    assert not (idx == np.arange(50)[:, None]).any() and idx.shape == (50, eng.k)
    assert eng.evaluate("calorie counter", category="FINANCE")["category"] == "FINANCE"
    print("engine ok", r["score"], r["verdict"], r["p_success"], d)


if __name__ == "__main__":
    demo()
