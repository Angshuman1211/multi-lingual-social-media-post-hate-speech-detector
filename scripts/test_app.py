"""Drive the Streamlit app headlessly and assert it actually predicts.

This runs the real app script through Streamlit's own script runner, which is
where the MuRIL meta-device failure appeared and where a plain import test would
not have caught it.

Run: python scripts/test_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

CASES = [
    ("you people are disgusting trash and should be wiped out", "Toxic"),
    ("Great match today, what a comeback by the team!", "Clean"),
    ("तुम लोग गंदे हो और तुम्हें यहाँ से निकाल देना चाहिए", "Toxic"),
    ("நீங்க எல்லாம் ஒரு பயனும் இல்லாதவங்க, இங்கிருந்து விரட்டணும்", "Toxic"),
    ("இந்த படம் மிகவும் அருமையாக இருந்தது, எனக்கு ரொம்ப பிடிச்சிருக்கு", "Clean"),
]


def run_case(model_label, text):
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=300)
    at.run()
    if at.exception:
        return None, f"startup: {at.exception[0].message}"

    at.radio[0].set_value(model_label).run()
    at.text_area(key="txt_post").set_value(text)
    at.button(key="btn_analyse").click().run()
    if at.exception:
        return None, f"on analyse: {at.exception[0].message}"

    blob = " ".join(m.value for m in at.markdown)
    verdict = "Toxic" if "&nbsp;Toxic" in blob else ("Clean" if "&nbsp;Clean" in blob else "?")
    prob = next((m.value for m in at.metric if "probability" in m.label.lower()), "?")
    return (verdict, prob), None


def main() -> int:
    failures = 0
    for model_label in ["TF-IDF + Logistic Regression", "MuRIL (fine-tuned)"]:
        print(f"\n=== {model_label} ===")
        for text, expected in CASES:
            got, err = run_case(model_label, text)
            if err:
                print(f"  FAIL  {text[:38]:<40} {err}")
                failures += 1
                continue
            verdict, prob = got
            ok = verdict == expected
            failures += 0 if ok else 1
            print(f"  {'ok  ' if ok else 'MISS'}  {text[:38]:<40} "
                  f"-> {verdict} (p={prob}, expected {expected})")

    # The example dropdown. This is the path that broke once: giving the text
    # area a key made Streamlit's session state own it, so the selectbox stopped
    # filling it in. Driving the text area directly hid that completely.
    print("\n=== example dropdown fills the box ===")
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=300)
    at.run()
    from app.app import EXAMPLES  # noqa: E402  the same dict the app renders

    for name in ["English - toxic - dehumanising", "Hindi - clean - weather",
                 "Bengali - toxic - violent", "Tamil - clean - film review"]:
        at.selectbox(key="example_pick").set_value(name).run()
        got = at.text_area(key="txt_post").value or ""
        ok = got.strip() == EXAMPLES[name].strip()
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<34} -> "
              f"{'loaded' if ok else f'box held {got[:28]!r}'}")

    # Live queue. This one really does hit the network, so a fetch returning
    # nothing is reported rather than failed.
    print("\n=== live moderation queue ===")
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=600)
    at.run()
    at.button(key="btn_fetch").click().run()
    if at.exception:
        print(f"  FAIL  live fetch raised: {at.exception[0].message}")
        failures += 1
    else:
        pulled = next((m.value for m in at.metric if m.label == "Posts pulled"), "0")
        flagged = next((m.value for m in at.metric if m.label == "Flagged for review"), "?")
        n = int(str(pulled).replace(",", "") or 0)
        print(f"  {'ok  ' if n > 0 else 'WARN'}  pulled {pulled} live posts, {flagged} flagged")
        if n == 0:
            print("        (no posts returned; the instances may be rate limiting)")

    # The other tabs render on every run, so a clean startup covers them.
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=300)
    at.run()
    print(f"\ntabs rendered without exception: {not at.exception}")
    print(f"dataframes on page: {len(at.dataframe)} | metrics: {len(at.metric)}")
    if at.exception:
        print(f"  {at.exception[0].message}")
        failures += 1

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
