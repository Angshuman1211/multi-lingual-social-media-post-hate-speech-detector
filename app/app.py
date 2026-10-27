"""Streamlit interface for the multilingual hate speech detector.

Run: streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from html import escape as html_escape

import altair as alt
import pandas as pd
import streamlit as st

from src import ingest
from src.config import DATASETS, LANGUAGES, RESULTS
from src.ingest import LEMMY_INSTANCES, MASTODON_INSTANCES
from src.predict import get_detector

SOURCE_CHOICES = MASTODON_INSTANCES + [f"lemmy:{i}" for i in LEMMY_INSTANCES]

# Palette roles, from the validated reference set. Categorical slots 1-3 for
# grouped metrics, blue/red as the diverging poles, and the fixed status pair
# for the verdict, which always ships with an icon and a word so colour never
# carries the meaning alone.
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
C_TOWARD_TOXIC = "#e34948"
C_TOWARD_CLEAN = "#2a78d6"
C_CRITICAL = "#d03b3b"
C_GOOD = "#0ca30c"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

st.set_page_config(page_title="Multilingual Hate Speech Detection",
                   page_icon="🛡️", layout="wide")

EXAMPLES = {
    "English - clean - sport": "Great match today, what a comeback by the team!",
    "English - clean - disagreement": "I disagree with the policy but I respect the argument",
    "English - toxic - dehumanising": "you people are disgusting trash and should be wiped out",
    "English - toxic - insult": "shut up you worthless idiot, nobody asked you",
    "English - borderline - xenophobic": "go back to your own country, we do not want your kind here",

    "Hindi - clean - weather": "आज मौसम बहुत अच्छा है और मैं बहुत खुश हूँ",
    "Hindi - clean - film review": "इस फिल्म की कहानी वाकई बहुत शानदार थी",
    "Hindi - toxic - exclusion": "तुम लोग गंदे हो और तुम्हें यहाँ से निकाल देना चाहिए",
    "Hindi - toxic - blame": "तुम्हारे जैसे लोगों की वजह से देश बर्बाद हो रहा है",

    "Bengali - clean - thanks": "তোমার কাজটা খুব সুন্দর হয়েছে, ধন্যবাদ",
    "Bengali - clean - weather": "আজকের আবহাওয়া সত্যিই চমৎকার ছিল",
    "Bengali - toxic - violent": "তোরা সব জানোয়ার, তোদের মেরে ফেলা উচিত",
    "Bengali - toxic - contempt": "এই লোকগুলো একেবারে অপদার্থ, এদের তাড়ানো দরকার",

    "Tamil - clean - film review": "இந்த படம் மிகவும் அருமையாக இருந்தது, எனக்கு ரொம்ப பிடிச்சிருக்கு",
    "Tamil - clean - weather": "இன்றைய வானிலை மிகவும் இனிமையாக இருந்தது",
    "Tamil - toxic - exclusion": "நீங்க எல்லாம் ஒரு பயனும் இல்லாதவங்க, இங்கிருந்து விரட்டணும்",
    "Tamil - toxic - blame": "உன்னை மாதிரி ஆளுங்களால தான் இந்த நாடு கெட்டு போச்சு",
}


def _load_example():
    """Push the picked example into the text area.

    The text area has a key, so Streamlit's session state owns its contents and
    a `value=` argument is ignored after the first render. Writing to session
    state from the selectbox callback is the only thing that actually updates it.
    """
    pick = st.session_state.get("example_pick", "(type my own)")
    st.session_state["txt_post"] = "" if pick == "(type my own)" else EXAMPLES[pick]


def chart_base(c):
    """Recessive chrome, so the marks carry the chart."""
    return c.configure_axis(
        grid=True, gridColor=GRID, gridWidth=1, domainColor=GRID,
        tickColor=GRID, labelColor=INK_MUTED, titleColor=INK_MUTED,
        labelFontSize=12, titleFontSize=12,
    ).configure_view(stroke=None).configure_legend(
        labelColor=INK_MUTED, titleColor=INK_MUTED, orient="top", direction="horizontal"
    )


@st.cache_resource
def detector():
    return get_detector()


@st.cache_data
def load_results(name):
    p = RESULTS / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_data_summary():
    p = ROOT / "data" / "processed" / "hate_speech_multilingual.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["lang", "label", "split", "source"])
    return (
        df.groupby("lang")
        .agg(rows=("label", "size"), toxic_share=("label", "mean"))
        .reset_index()
    )


det = detector()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## Model")
    has_muril = det.muril_available()
    choice = st.radio(
        "Classifier",
        ["TF-IDF + Logistic Regression", "MuRIL (fine-tuned)"],
        index=1 if has_muril else 0,
        help="The linear model is the baseline. MuRIL is a transformer pretrained "
             "on 17 Indian languages.",
    )
    model = "muril" if choice.startswith("MuRIL") else "baseline"
    if model == "muril" and not has_muril:
        st.warning("MuRIL is not trained yet. Run `python -m src.transformer`. "
                   "Falling back to the baseline.")
        model = "baseline"

    st.markdown("---")
    st.markdown("## Decision threshold")
    st.caption(
        "Each language gets its own threshold, tuned on validation data. "
        "Override it to trade precision against recall."
    )
    override = st.checkbox("Override tuned threshold", value=False)
    manual_thr = st.slider("Threshold", 0.05, 0.95, 0.50, 0.01, disabled=not override)

    st.markdown("---")
    st.caption(
        "Languages: English, Hindi, Bengali, Tamil. The language is detected "
        "from the script, so you can paste any of the four straight in."
    )

st.title("Multilingual Hate Speech Detection")
st.caption("English · Hindi · Bengali · Tamil")

tab_live, tab_analyse, tab_batch, tab_perf, tab_data = st.tabs(
    ["Live moderation queue", "Analyse text", "Batch (CSV)",
     "Model performance", "Data and limits"]
)

# ---------------------------------------------------------------- live queue
with tab_live:
    st.markdown(
        "Pulls **live public posts** from Mastodon and Lemmy, scores every one, and "
        "ranks them the way a moderation queue would. No API key is involved: these "
        "are the platforms that still serve public timelines anonymously, now that "
        "Reddit returns 403 and X charges for access."
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sources = st.multiselect(
            "Sources",
            SOURCE_CHOICES,
            default=["mastodon.world", "fosstodon.org", "lemmy:lemmy.world"],
        )
    with c2:
        tag = st.text_input("Hashtag (optional)", placeholder="politics",
                            help="Public timelines skew benign. A hashtag gives the "
                                 "queue something to triage.")
    with c3:
        per_source = st.number_input("Posts per source", 5, 40, 20, step=5)

    if st.button("Fetch and score", type="primary", key="btn_fetch"):
        with st.spinner("Pulling live posts..."):
            posts, errors = ingest.fetch(sources, per_source=int(per_source), tag=tag)
        for e in errors:
            st.warning(f"source unavailable: {e}")

        if not posts:
            st.info("No posts came back. Try different sources or drop the hashtag.")
        else:
            with st.spinner(f"Scoring {len(posts)} posts..."):
                scored = det.predict_batch([p["text"] for p in posts], model=model)
            feed = pd.DataFrame(posts)
            feed["probability"] = scored["probability"]
            feed["flagged"] = scored["label"].astype(bool)
            feed["language"] = scored["language"]
            feed["age_min"] = feed["created_at"].map(ingest.age_minutes)
            feed = feed.sort_values("probability", ascending=False)
            st.session_state["feed"] = feed
            st.session_state["feed_model"] = scored["model"]

    feed = st.session_state.get("feed")
    if feed is not None:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Posts pulled", f"{len(feed):,}")
        k2.metric("Flagged for review", int(feed["flagged"].sum()))
        k3.metric("Share flagged", f"{feed['flagged'].mean():.1%}")
        newest = feed["age_min"].min()
        k4.metric("Newest post", "unknown" if pd.isna(newest) else f"{newest:.0f} min old")
        st.caption(f"Scored with {st.session_state.get('feed_model', model)}.")

        langs = feed["language"].map(LANGUAGES).fillna(feed["language"]).value_counts()
        if len(langs) and langs.index[0] == "English" and langs.iloc[0] / len(feed) > 0.6:
            st.info(
                "These instances are overwhelmingly English, so the Hindi, Bengali and "
                "Tamil models rarely get to fire on live data. That is a property of "
                "the open fediverse, not of the detector.",
                icon="ℹ",
            )

        st.markdown("#### Queue, most toxic first")
        for _, row in feed.head(25).iterrows():
            colour = C_CRITICAL if row["flagged"] else INK_MUTED
            label = "FLAGGED" if row["flagged"] else "ok"
            age = "" if pd.isna(row["age_min"]) else f" · {row['age_min']:.0f} min ago"
            st.markdown(
                f"""
                <div style="border-left:4px solid {colour};padding:6px 12px;margin-bottom:8px;
                            background:#f9f9f7;border-radius:4px;">
                  <div style="font-size:12px;color:{colour};font-weight:700;">
                    {label} &nbsp;{row['probability']:.3f}
                  </div>
                  <div style="font-size:12px;color:{INK_MUTED};">
                    {row['author']} · {row['platform']} ·
                    {LANGUAGES.get(row['language'], row['language'])}{age}
                  </div>
                  <div style="margin-top:4px;">{html_escape(row['text'][:400])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.download_button(
            "Download the scored queue",
            feed[["text", "author", "platform", "language", "probability", "flagged", "url"]]
            .to_csv(index=False).encode("utf-8"),
            file_name="live_moderation_queue.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------- analyse
with tab_analyse:
    col_in, col_out = st.columns([3, 2], gap="large")

    with col_in:
        st.selectbox(
            "Load an example",
            ["(type my own)"] + list(EXAMPLES),
            key="example_pick",
            on_change=_load_example,
            help="Four or five samples per language, clean and toxic, plus a "
                 "borderline English case the model gets wrong.",
        )
        text = st.text_area(
            "Post to check", height=160, key="txt_post",
            placeholder="Paste a social media post in any of the four languages",
        )
        go = st.button("Analyse", type="primary", width="stretch", key="btn_analyse")

    if go and text.strip():
        res = det.predict(text, model=model)
        if override:
            res["threshold"] = manual_thr
            res["label"] = int(res["probability"] >= manual_thr)
            res["label_name"] = "Toxic" if res["label"] else "Clean"

        with col_out:
            toxic = res["label"] == 1
            colour = C_CRITICAL if toxic else C_GOOD
            icon = "⚠" if toxic else "✓"
            st.markdown(
                f"""
                <div style="border:1px solid {colour};border-left:6px solid {colour};
                            border-radius:8px;padding:14px 18px;margin-bottom:12px;">
                  <div style="font-size:26px;font-weight:700;color:{colour};">
                    {icon}&nbsp;{res['label_name']}
                  </div>
                  <div style="color:{INK_MUTED};font-size:13px;margin-top:2px;">
                    {res['language_name']} detected &middot; {res['model']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            a, b = st.columns(2)
            a.metric("Toxic probability", f"{res['probability']:.3f}")
            b.metric("Threshold used", f"{res['threshold']:.2f}")

            if "en3" in res:
                st.markdown("**Davidson category (English only)**")
                d3 = pd.DataFrame(
                    {"category": list(res["en3"]), "probability": list(res["en3"].values())}
                )
                st.dataframe(
                    d3.sort_values("probability", ascending=False),
                    hide_index=True, width="stretch",
                    column_config={"probability": st.column_config.ProgressColumn(
                        "probability", min_value=0.0, max_value=1.0, format="%.3f")},
                )
                if res["en3_top"] == "neither" and toxic:
                    st.caption(
                        "The three-way model is trained on Davidson alone, which keys on "
                        "explicit slurs. It disagrees here, which is exactly the blind spot "
                        "the second English corpus was added to cover."
                    )

        st.markdown("---")
        st.markdown("#### Which words drove the decision")
        st.caption(
            "I remove one word at a time and measure how far the toxic probability moves. "
            "Red pushes toward toxic, blue pushes toward clean."
        )
        contrib = det.explain(text, model=model, top_k=10)
        if contrib:
            cdf = pd.DataFrame(contrib, columns=["word", "effect"])
            cdf["direction"] = cdf["effect"].apply(
                lambda v: "toward toxic" if v > 0 else "toward clean"
            )
            ch = (
                alt.Chart(cdf)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("effect:Q", title="change in toxic probability"),
                    # Step sizes the row band itself. A fixed bar height instead
                    # let the bars overrun their bands and merge into one block.
                    y=alt.Y("word:N", sort=alt.EncodingSortField("effect", order="descending"),
                            title=None, scale=alt.Scale(paddingInner=0.25),
                            axis=alt.Axis(labelOverlap=False, labelLimit=160)),
                    color=alt.Color(
                        "direction:N",
                        scale=alt.Scale(domain=["toward toxic", "toward clean"],
                                        range=[C_TOWARD_TOXIC, C_TOWARD_CLEAN]),
                        legend=alt.Legend(title=None),
                    ),
                    tooltip=[alt.Tooltip("word:N"), alt.Tooltip("effect:Q", format="+.4f")],
                )
                .properties(height=alt.Step(30))
            )
            st.altair_chart(chart_base(ch), width="stretch")
        else:
            st.info("Not enough words to attribute.")
    elif go:
        st.warning("Enter some text first.")

# ---------------------------------------------------------------- batch
with tab_batch:
    st.markdown("Upload a CSV of posts and score every row at once.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        st.write(f"{len(df_in):,} rows loaded.")
        col = st.selectbox("Column holding the text", list(df_in.columns))
        if st.button("Score all rows", type="primary", key="btn_batch"):
            with st.spinner("Scoring..."):
                out = det.predict_batch(df_in[col].astype(str).tolist(), model=model)
            res_df = pd.DataFrame({
                "text": out["text"],
                "language": out["language"],
                "toxic_probability": out["probability"].round(4),
                "prediction": ["Toxic" if v else "Clean" for v in out["label"]],
            })
            st.success(f"Scored with {out['model']}.")

            k1, k2 = st.columns(2)
            k1.metric("Flagged toxic", f"{int(out['label'].sum()):,}")
            k2.metric("Share flagged", f"{out['label'].mean():.1%}")

            by_lang = (
                res_df.assign(flag=(res_df.prediction == "Toxic").astype(int))
                .groupby("language")
                .agg(posts=("flag", "size"), flagged=("flag", "mean"))
                .reset_index()
            )
            by_lang["language"] = by_lang["language"].map(LANGUAGES).fillna(by_lang["language"])
            ch = (
                alt.Chart(by_lang)
                .mark_bar(cornerRadiusEnd=4, size=28, color=C_SERIES[0])
                .encode(
                    x=alt.X("flagged:Q", title="share flagged toxic",
                            axis=alt.Axis(format="%")),
                    y=alt.Y("language:N", sort="-x", title=None),
                    tooltip=[alt.Tooltip("language:N"), alt.Tooltip("posts:Q"),
                             alt.Tooltip("flagged:Q", format=".1%")],
                )
                .properties(height=max(120, 34 * len(by_lang)))
            )
            st.altair_chart(chart_base(ch), width="stretch")
            st.dataframe(res_df, width="stretch", hide_index=True)
            st.download_button(
                "Download scored CSV",
                res_df.to_csv(index=False).encode("utf-8"),
                file_name="hate_speech_predictions.csv",
                mime="text/csv",
            )
    else:
        st.caption("The file needs one column of text. Everything else is ignored.")

# ---------------------------------------------------------------- performance
with tab_perf:
    base = load_results("baseline_results.json")
    muril = load_results("muril_all_results.json")
    transfer = load_results("muril_no_ta_results.json")

    if base is None:
        st.info("No results yet. Run `python -m src.baseline`.")
    else:
        rows = []
        for lang in ["en", "hi", "bn", "ta"]:
            if lang in base.get("per_language", {}):
                m = base["per_language"][lang]
                rows.append({"language": LANGUAGES[lang], "model": "TF-IDF + LogReg",
                             "macro F1": m["macro_f1"], "toxic precision": m["toxic_precision"],
                             "toxic recall": m["toxic_recall"], "ROC AUC": m.get("roc_auc")})
            if muril and lang in muril.get("test_tuned_threshold", {}):
                m = muril["test_tuned_threshold"][lang]
                rows.append({"language": LANGUAGES[lang], "model": "MuRIL",
                             "macro F1": m["macro_f1"], "toxic precision": m["toxic_precision"],
                             "toxic recall": m["toxic_recall"], "ROC AUC": m.get("roc_auc")})
        perf = pd.DataFrame(rows)

        st.markdown("#### Macro F1 on the held-out test set")
        st.caption("Macro F1 over both classes, so the majority class cannot carry the score.")
        ch = (
            alt.Chart(perf)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("model:N", title=None, axis=alt.Axis(labels=False, ticks=False),
                        scale=alt.Scale(paddingInner=0.18)),
                y=alt.Y("macro F1:Q", title="macro F1", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("model:N",
                                scale=alt.Scale(range=C_SERIES[: perf["model"].nunique()]),
                                legend=alt.Legend(title=None)),
                column=alt.Column("language:N", title=None,
                                  header=alt.Header(labelColor=INK_MUTED)),
                tooltip=["language:N", "model:N", alt.Tooltip("macro F1:Q", format=".3f")],
            )
            .properties(width=120, height=240)
        )
        st.altair_chart(chart_base(ch))

        st.markdown("#### Every metric")
        st.caption("The table is the accessible view of the chart above.")
        st.dataframe(
            perf.style.format({c: "{:.3f}" for c in
                               ["macro F1", "toxic precision", "toxic recall", "ROC AUC"]}),
            width="stretch", hide_index=True,
        )

        if muril is None:
            st.info("MuRIL results appear here once `python -m src.transformer` has run.")

        ci = load_results("bootstrap_ci.json")
        if ci:
            st.markdown("#### Is the transformer actually better?")
            st.caption(
                "The gap per language, bootstrapped over 2,000 resamples of the test "
                "set. A point estimate on 735 Hindi rows would not mean much on its own."
            )
            cdf = pd.DataFrame(ci)
            chart = (
                alt.Chart(cdf)
                .mark_bar(cornerRadiusEnd=4, color=C_SERIES[0])
                .encode(
                    x=alt.X("observed_delta:Q", title="MuRIL macro F1 minus baseline"),
                    y=alt.Y("language:N", sort="-x", title=None,
                            scale=alt.Scale(paddingInner=0.25)),
                    tooltip=["language:N", alt.Tooltip("observed_delta:Q", format="+.3f"),
                             alt.Tooltip("ci_low:Q", format="+.3f"),
                             alt.Tooltip("ci_high:Q", format="+.3f")],
                )
                .properties(height=alt.Step(34))
            )
            band = (
                alt.Chart(cdf)
                .mark_rule(size=2, color=INK_MUTED)
                .encode(x="ci_low:Q", x2="ci_high:Q", y=alt.Y("language:N", sort="-x"))
            )
            zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(
                color="#0b0b0b", strokeDash=[4, 3]).encode(x="z:Q")
            st.altair_chart(chart_base(chart + band + zero), width="stretch")
            show = cdf[["language", "n", "observed_delta", "ci_low", "ci_high", "significant"]]
            st.dataframe(
                show.style.format({"observed_delta": "{:+.3f}", "ci_low": "{:+.3f}",
                                   "ci_high": "{:+.3f}", "n": "{:,}"}),
                width="stretch", hide_index=True,
            )
            st.caption(
                "MuRIL is significantly ahead only on English and Bengali, the two "
                "largest corpora. On Hindi and Tamil the interval crosses zero, so "
                "neither model can be called better there."
            )

        if transfer:
            st.markdown("#### Zero-shot transfer into Tamil")
            st.caption(
                "This model never saw a Tamil training example. It is scored on the "
                "same Tamil test set. Compare against the in-language model above: the "
                "ranking carries over, the decision threshold does not."
            )
            t = transfer["test_default_threshold"].get("ta")
            if t:
                in_lang = (muril or {}).get("test_tuned_threshold", {}).get("ta", {})
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "ROC AUC (zero-shot)", f"{t.get('roc_auc', float('nan')):.3f}",
                    delta=(None if not in_lang else
                           f"{t.get('roc_auc', 0) - in_lang.get('roc_auc', 0):+.3f} vs trained on Tamil"),
                )
                c2.metric("macro F1 (zero-shot)", f"{t['macro_f1']:.3f}")
                c3.metric(
                    "toxic recall", f"{t['toxic_recall']:.3f}",
                    delta=(None if not in_lang else
                           f"{t['toxic_recall'] - in_lang.get('toxic_recall', 0):+.3f} vs trained on Tamil"),
                )
                st.caption(
                    "AUC 0.816 with no Tamil data at all is 82% of the way from chance "
                    "to the in-language model, across language families. Recall collapses "
                    "to 9% only because there is no Tamil validation set to calibrate the "
                    "threshold on."
                )

        st.markdown("#### English, three-way")
        st.caption("Davidson separates hate speech from merely offensive language. "
                   "The hate class is the hard one, and this is the usual result for it.")
        st.code(base["en3"]["report"], language=None)

# ---------------------------------------------------------------- data
with tab_data:
    summary = load_data_summary()
    st.markdown("#### Corpora")
    src = pd.DataFrame([
        {"language": LANGUAGES.get(k.split("_")[0], k), "source": v["hf_id"],
         "reference": v["citation"]}
        for k, v in DATASETS.items()
    ])
    st.dataframe(src, width="stretch", hide_index=True)

    if summary is not None:
        summary["language"] = summary["lang"].map(LANGUAGES)
        st.markdown("#### Rows and class balance after cleaning")
        ch = (
            alt.Chart(summary)
            .mark_bar(cornerRadiusEnd=4, size=28, color=C_SERIES[0])
            .encode(
                x=alt.X("toxic_share:Q", title="share labelled toxic",
                        axis=alt.Axis(format="%")),
                y=alt.Y("language:N", sort="-x", title=None),
                tooltip=["language:N", alt.Tooltip("rows:Q", format=","),
                         alt.Tooltip("toxic_share:Q", format=".1%")],
            )
            .properties(height=140)
        )
        st.altair_chart(chart_base(ch), width="stretch")
        st.dataframe(
            summary[["language", "rows", "toxic_share"]].style.format(
                {"rows": "{:,}", "toxic_share": "{:.1%}"}),
            width="stretch", hide_index=True,
        )

    st.markdown("#### What this system cannot do")
    st.markdown(
        """
- **Tamil keeps only a fifth of its corpus.** The Tamil YouTube comments are 81%
  romanised, and script routing cannot see those as Tamil, so I train on the 7,357
  comments actually written in Tamil script. Code-mixed Tamil is out of scope here
  even though it is how much of Tamil social media is really written.
- **Each language is labelled by different people under different guidelines.** I
  pooled four corpora into one toxic / clean space, which is the only label they
  share. A Hindi `HOF` tag and a Tamil `Offensive_Targeted_Insult` tag do not mean
  quite the same thing.
- **Romanised Hindi, Bengali and Tamil route to English.** Detection is script based,
  so `tum log bekar ho` is scored by the English model, which never trained on it.
- **The live queue is mostly English.** Mastodon and Lemmy skew heavily to English
  and European languages, so the three Indic models rarely fire on live data.
- **The models reflect their training data, including its bias.** Davidson in
  particular over-flags African American English, a documented result. Adding the
  Berkeley corpus reduces the slur-keying but does not remove the bias.
- **This is coursework, not a moderation system.** It should never decide anything
  about a real account on its own.
        """
    )
