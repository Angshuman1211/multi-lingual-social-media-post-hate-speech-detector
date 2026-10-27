# Multilingual Hate Speech Detection

Hate speech detection for social media posts in **English, Hindi, Bengali and Tamil**,
with a web interface that scores **live posts pulled from real platforms**. I compare a
classical TF-IDF baseline against a fine-tuned MuRIL transformer and report per-language
results rather than one pooled number.

NLP course project, Semester 7.

---

## Two problems I ran into, and what they changed

**English was a slur detector, not a hate detector.** Davidson on its own scored "you
people are disgusting trash and should be wiped out" at 0.07 toxic, and leave-one-out
attribution showed that *removing* the word "trash" made the text look more toxic.
Adding the Berkeley corpus, which carries plenty of hate containing no slur, fixed that
example to 0.93 and dropped English macro F1 from 0.92 to 0.83. The lower number is the
honest one, because the first task was easier than the real one.

**The original Telugu corpus was not social media, and its labels were wrong.** I built
the first version on the IIIT-H Telugu set, then audited it: the rows are scraped news
prose, not posts, it is only 1.1% positive, and sampling the hate-labelled rows turns up
text that is plainly not hate speech, including a sentence about India's S-400 missile
procurement deal. Its apparent AUC of 0.927 was therefore at risk of measuring news
topic rather than hate. I replaced it with **Tamil**, which is YouTube comments from the
DravidianCodeMix shared task:

| | old Telugu | new Tamil |
|---|---:|---:|
| source | scraped news prose | YouTube comments |
| usable rows | 34,999 | 7,266 |
| toxic share | 1.1% | **19.7%** |
| positives | ~380 | **1,432** |
| positives in test | 55 | **214** |

Smaller, but 3.8× more positives at 18× better balance, from actual social media, with
labels that survive inspection. Marathi would have been the more obvious Indian language
to pick, but it shares Devanagari with Hindi and script-based routing cannot separate
the two.

## Data

| Language | Corpus | Platform | Rows (clean) | Toxic share |
|---|---|---|---:|---:|
| English | Davidson et al. (2017) | Twitter | 24,574 | 83.3% |
| English | Berkeley MHS, Kennedy et al. (2020) | YouTube, Reddit, Twitter | 39,536 | 26.3% |
| Hindi | HASOC, Mandl et al. (2019) | Twitter | 4,891 | 51.4% |
| Bengali | Karim et al. (2020) | Facebook, YouTube | 29,717 | 33.5% |
| Tamil | DravidianCodeMix, Chakravarthi et al. (2021) | YouTube | 7,266 | 19.7% |

Every corpus is platform-sourced, which is the point of a social media project. 61% of
the Davidson rows still carry `@user` or URL markers, and my preprocessing preserves
that register rather than scrubbing it: mentions become `<user>`, URLs become `<url>`,
hashtags keep their word, emoji survive.

**105,984 rows** after cleaning and cross-source deduplication, split 70/15/15,
stratified jointly on language and label so every language keeps its toxic ratio in
all three splits.

The Tamil corpus is 81% romanised code-mixed text. Script-based routing cannot see those
rows as Tamil, so I keep only the 7,357 comments actually written in Tamil script and
treat code-mixing as out of scope.

Every corpus uses a different label scheme, so I pooled them into the only space they
share: **toxic vs clean**. English keeps a second three-way model (hate / offensive /
neither) because Davidson is the only corpus that draws that line.

## Models

**Baseline.** TF-IDF into logistic regression, with word 1-2 grams *and* character
2-5 grams. The character features matter: Bengali and Tamil are morphologically rich
enough that a word vocabulary misses most inflected forms, and character n-grams also
absorb deliberate misspelling. Trained per language and jointly, with balanced class
weights.

**Transformer.** `google/muril-base-cased`, fine-tuned. MuRIL is pretrained on 17
Indian languages plus English, so it reads Tamil and Bengali script natively rather
than incidentally. Class-weighted loss, 2 epochs, max length 128, on the Mac GPU via
MPS.

I freeze the word embedding matrix. MuRIL's vocabulary is 197k, so that matrix is 151M
of the model's 238M parameters, and freezing it cut the step cost from 2.25s to 0.90s
with no reason to expect a loss on 75k training examples. Max length stays at 128
because Hindi needs it: median Hindi post is 100 tokens where English is 26, so
truncating at 64 would cut most Hindi posts in half.

**Thresholds.** I tune the decision threshold per language on validation data instead
of assuming 0.5, because at 1% prevalence argmax almost never fires even when the
ranking is good.

## Results

Macro F1 on the held-out test set, per language, each at its own tuned threshold.

| Language | TF-IDF + LogReg | MuRIL | Delta | AUC baseline | AUC MuRIL |
|---|---:|---:|---:|---:|---:|
| English | 0.831 | **0.854** | +0.023 | 0.906 | 0.931 |
| Hindi | **0.735** | 0.706 | -0.029 | 0.815 | 0.775 |
| Bengali | 0.879 | **0.895** | +0.016 | 0.950 | 0.965 |
| Tamil | 0.732 | **0.750** | +0.018 | 0.855 | 0.884 |
| mean | 0.794 | 0.801 | +0.007 | | |

### The transformer only wins where the data is large

A point estimate on 735 Hindi test rows is not evidence of anything, so I bootstrapped
the per-language gap over 2,000 resamples of the test set.

| Language | test rows | delta | 95% CI | verdict |
|---|---:|---:|---|---|
| English | 9,617 | +0.023 | [+0.017, +0.030] | **significant** |
| Bengali | 4,457 | +0.016 | [+0.006, +0.025] | **significant** |
| Tamil | 1,089 | +0.018 | [-0.012, +0.047] | not significant |
| Hindi | 735 | -0.029 | [-0.063, +0.007] | not significant |

So the honest claim is narrower than "MuRIL wins": **MuRIL beats the linear baseline
on the two largest corpora and is indistinguishable from it on the two smallest.**
Hindi looks like a loss and is ahead in only 6% of resamples, but the interval crosses
zero, so I am not claiming the baseline is better there either. The transformer's
advantage tracks training volume, which is the useful lesson: at a few thousand labelled
examples, 238M pretrained parameters buy nothing a TF-IDF model cannot match.

This also corrects an earlier version of this project. Before I audited the data, the
fourth language was Telugu and MuRIL appeared to win on all four by a wider mean margin
of +0.025. Most of that margin came from a Telugu gain of +0.058 on a corpus that turned
out to be mislabelled news prose. Fixing the data removed the headline.

### Cross-lingual transfer mostly works, and I had this backwards

I trained the same model on English, Hindi and Bengali only, held Tamil out entirely,
and scored it on the identical Tamil test set.

| Tamil test | trained with Tamil | trained without Tamil |
|---|---:|---:|
| ROC AUC | 0.884 | **0.816** |
| macro F1 | 0.750 | 0.526 |
| toxic recall | 0.631 | 0.093 |

A model that has never seen a Tamil training example still reaches AUC 0.816, which is
82% of the way from chance to the in-language model. Most of the ability to *rank* Tamil
posts by toxicity carries over from three unrelated languages, and it crosses language
families to do it, since Tamil is Dravidian while Hindi and Bengali are Indo-Aryan and
English is Germanic.

What does not carry over is the **threshold**. At the default 0.5 the held-out model
catches 9% of toxic Tamil posts against 63% for the in-language model, because there is
no Tamil validation data to calibrate on. The ordering transfers, the decision boundary
does not. That is the practical finding: for a new language you can borrow a ranker for
free, but you still need labelled data to decide where to cut.

**This reverses what an earlier version of this project concluded.** Built on the Telugu
corpus, the same experiment gave AUC 0.541, which is chance, and I wrote that nothing
transfers and blamed the Indo-Aryan versus Dravidian split. That was an artifact of a
corpus whose labels do not survive inspection. The language-family explanation was wrong
as well, since transfer into Tamil crosses exactly the same divide and works.

**English three-way** (Davidson only): macro F1 0.742, with the hate class at F1 0.430
against 0.929 for offensive. The hate-versus-offensive boundary is the hard part, and
that gap is the usual result on this corpus.

Full JSON in `results/`, and the same tables render in the app's **Model performance**
tab. Regenerate with `python -m src.baseline`, `python -m src.transformer` and
`python scripts/bootstrap_ci.py`.

## Running it

If the models are already trained, this is the whole thing:

```bash
.venv/bin/streamlit run app/app.py       # then open http://localhost:8501
```

Use `.venv/bin/streamlit`, not a bare `streamlit`. The system Python on this machine has
a broken scipy, which is the reason the venv exists.

### From scratch

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python scripts/fetch_muril.py    # MuRIL weights, ~950 MB, resumable
.venv/bin/python -m src.data_prep          # download and pool the corpora      ~3 min
.venv/bin/python -m src.baseline           # TF-IDF models                      ~1 min
.venv/bin/python -m src.transformer        # fine-tune MuRIL                   ~65 min
.venv/bin/python scripts/bootstrap_ci.py   # confidence intervals              ~10 min

.venv/bin/python -m src.transformer --exclude-lang ta   # transfer experiment  ~65 min
```

Training times are for the Mac GPU through MPS. The app works without the transformer;
it falls back to the baseline and says so in the sidebar.

### Checking nothing broke

```bash
.venv/bin/python scripts/test_app.py       # should print ALL PASS
```

This drives the real app through Streamlit's script runner: it sets widgets, clicks
buttons, loads examples from the dropdown, and fetches live posts. Import-level tests
would not catch the failures that actually happened here, both of which were only
visible through the running interface.

### Why `fetch_muril.py` exists

MuRIL publishes `pytorch_model.bin` with no safetensors, so `transformers` 5.x refuses
to load it and the venv pins 4.57. On top of that the Hugging Face Xet transfer failed
outright on my connection and `curl -C -` stalled at 35%. The script pulls the file in
8 MB ranges, retries each one, and verifies the published SHA-256.

## Layout

```
src/config.py           paths, label space, corpus registry, script ranges
src/preprocess.py       text cleaning and script-based language detection
src/data_prep.py        download, unify, deduplicate, split
src/baseline.py         TF-IDF + logistic regression
src/transformer.py      MuRIL fine-tuning and evaluation
src/evaluate.py         metrics, threshold tuning, reports
src/predict.py          inference wrapper the app calls
src/ingest.py           live post ingestion from Mastodon and Lemmy
app/app.py              Streamlit interface, five tabs
scripts/fetch_muril.py  resumable, checksum-verified weight download
scripts/bootstrap_ci.py confidence intervals on the baseline vs MuRIL gap
scripts/test_app.py     drives the app headlessly, prints ALL PASS
data/sample_posts.csv   20 posts across the four languages, for the batch tab
results/                every metric as JSON, read by the app
```

## Interface

Five tabs: a live moderation queue, analyse a single post, score a CSV in bulk, inspect
model performance, and read the data and its limits. Language is detected from the
Unicode script, so any of the four can be pasted straight in.

### Trying it

The **Analyse text** tab has a dropdown of worked examples, four or five per language,
clean and toxic, including a borderline English case (*"go back to your own country, we
do not want your kind here"*) that the model scores below threshold and therefore misses.
Switching the sidebar between the two classifiers on the same sentence is the quickest
way to see where they differ.

### The live moderation queue

The queue pulls **real public posts** and ranks them the way a moderation tool would,
most toxic first. No API key is needed. Reddit now returns 403 to anonymous JSON,
Bluesky's public endpoint 403s, and X charges for access, so the sources are the
platforms that still serve public timelines openly: **Mastodon** (mastodon.world,
fosstodon.org, techhub.social, mas.to) and **Lemmy** (lemmy.world). An optional hashtag
switches Mastodon to its tag timeline, because public timelines skew benign and give the
queue little to triage.

This turned out to be the most informative part of the project. On a 50-post live pull
the model flagged 3, and **all three were false positives**: a snarky voting quip, an
automated aviation ACARS telemetry message, and a Wall Street opinion post. None were
hate speech. A test set reports 0.83 macro F1 for English, but live precision is visibly
poor, because real feeds are dominated by ordinary text that the training corpora never
represented. That gap only becomes visible on live data.

Explanations use **leave-one-token-out occlusion**: drop each word, measure how far the
toxic probability moves. It is model agnostic, so the interface explains the linear
model and the transformer the same way.

## Limits

- **Tamil keeps only a fifth of its corpus**, because 81% of it is romanised code-mixed
  text that script routing cannot recognise as Tamil.
- **Labels are not commensurable.** A Hindi `HOF` tag and a Bengali `hate` tag were
  written by different annotators under different guidelines.
- **Romanised Hindi, Bengali and Tamil route to English**, because detection is script
  based. `tum log bekar ho` gets scored by a model that never trained on it.
- **The live queue is mostly English.** The open fediverse skews heavily English and
  European, so the three Indic models rarely fire on live data.
- **Live precision is poor**, as the queue itself shows. Benchmark F1 does not transfer
  to an unfiltered real feed.
- **The models inherit their corpora's bias.** Davidson over-flags African American
  English, which is documented. The Berkeley corpus reduces the slur-keying but does
  not remove the bias.
- This is coursework. It should not decide anything about a real account on its own.

## References

Davidson, T., Warmsley, D., Macy, M., Weber, I. (2017). Automated Hate Speech Detection
and the Problem of Offensive Language. *ICWSM*.

Kennedy, C. J., Bacon, G., Sahn, A., von Vacano, C. (2020). Constructing Interval
Variables via Faceted Rasch Measurement and Multitask Deep Learning.

Mandl, T. et al. (2019). Overview of the HASOC Track at FIRE 2019.

Karim, M. R. et al. (2020). Classification Benchmarks for Under-resourced Bengali
Language based on Multichannel Convolutional-LSTM Network.

Chakravarthi, B. R. et al. (2021). DravidianCodeMix: Sentiment Analysis and Offensive
Language Identification Dataset for Dravidian Languages in Code-Mixed Text.

Khanuja, S. et al. (2021). MuRIL: Multilingual Representations for Indian Languages.
