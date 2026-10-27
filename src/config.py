"""Central paths, label space and dataset registry."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

for _d in (DATA_RAW, DATA_PROC, MODELS, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Binary space shared by all four languages. The source corpora disagree on
# everything finer than this, so this is the only label I can honestly pool.
LABELS = {0: "clean", 1: "toxic"}

# English only. Davidson separates hate from merely offensive, which is the
# distinction the binary head throws away, so I keep it as a second model.
LABELS_EN3 = {0: "hate speech", 1: "offensive", 2: "neither"}

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
}

# Unicode ranges I use for script-based language routing in the app.
# Marathi would have been the obvious fourth language but it shares Devanagari
# with Hindi, so script alone cannot separate them.
SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)],  # Devanagari
    "bn": [(0x0980, 0x09FF)],  # Bengali
    "ta": [(0x0B80, 0x0BFF)],  # Tamil
}

DATASETS = {
    "en": {
        "hf_id": "tdavidson/hate_speech_offensive",
        "citation": "Davidson et al. (2017), ICWSM",
        "text_col": "tweet",
        "label_col": "class",
    },
    # Davidson on its own learns to spot slurs rather than hate, so English gets
    # a second corpus that contains plenty of hate carrying no slur at all.
    "en_mhs": {
        "hf_id": "ucberkeley-dlab/measuring-hate-speech",
        "citation": "Kennedy et al. (2020), Measuring Hate Speech",
        "text_col": "text",
        "label_col": "hate_speech_score",
    },
    "hi": {
        "hf_id": "nikitadesai/hasoc",
        "citation": "HASOC (Mandl et al., 2019), Hindi subtask",
        "text_col": "rawtweets",
        "label_col": "binary class",
    },
    "bn": {
        "hf_id": "sadi2oo3q/Bengali_hate_speech_dataset",
        "citation": "Karim et al. (2020), Bengali Hate Speech Dataset",
        "text_col": "sentence",
        "label_col": "hate",
    },
    # Replaced the Telugu corpus here. That one was scraped news prose rather
    # than social media, only 1.1% positive, and sampling its positives turned up
    # rows that are plainly not hate speech (an S-400 missile procurement
    # headline among them). This Tamil set is YouTube comments, which is what
    # the rest of the project is built on.
    "ta": {
        "hf_id": "community-datasets/offenseval_dravidian",
        "citation": "Chakravarthi et al. (2021), DravidianCodeMix, Tamil YouTube comments",
        "text_col": "text",
        "label_col": "label",
    },
}

# Tamil arrives heavily romanised. Script routing cannot see those rows as
# Tamil, so I keep only rows that are meaningfully in Tamil script.
MIN_NATIVE_SCRIPT = 0.2

# MuRIL is pretrained on 17 Indian languages plus English and sees Tamil and
# Bengali script natively, which XLM-R only covers incidentally.
# It publishes no safetensors, so scripts/fetch_muril.sh pulls the weights to a
# local folder and I prefer that copy when it is present.
_LOCAL_MURIL = MODELS / "muril-base-cased"
TRANSFORMER_MODEL = (
    str(_LOCAL_MURIL)
    if (_LOCAL_MURIL / "pytorch_model.bin").exists()
    else "google/muril-base-cased"
)
MAX_LEN = 128
