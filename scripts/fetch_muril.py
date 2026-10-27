"""Fetch the MuRIL weights over a connection that keeps dropping.

MuRIL publishes no safetensors, the Xet transfer failed outright, and plain
`curl -C -` stalled at 35%. Range requests do work, so I pull the file in fixed
chunks, retry each one on its own, and verify the sha256 the hub advertises.

Run: python scripts/fetch_muril.py
"""

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/google/muril-base-cased/resolve/main/pytorch_model.bin"
SHA256 = "e65e3b7aaa3ce16d23fc4218783efd0417880655f8f398e6cb1a111a0de0c95f"
TOTAL = 953477430
CHUNK = 8 << 20  # 8 MB keeps each request short enough to usually survive.
DEST = Path(__file__).resolve().parent.parent / "models" / "muril-base-cased"
SMALL = ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"]
BASE = "https://huggingface.co/google/muril-base-cased/resolve/main"


def fetch_range(start: int, end: int, attempts: int = 12) -> bytes:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    for a in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status not in (200, 206):
                    raise OSError(f"status {r.status}")
                data = r.read()
            if len(data) == end - start + 1:
                return data
            raise OSError(f"short read {len(data)} != {end - start + 1}")
        except Exception as e:  # noqa: BLE001 - any network failure retries
            if a == attempts:
                raise
            time.sleep(min(2 * a, 20))
    raise OSError("unreachable")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in SMALL:
        p = DEST / name
        if not p.exists() or p.stat().st_size == 0:
            with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as r:
                p.write_bytes(r.read())
            print(f"got {name}")

    target = DEST / "pytorch_model.bin"
    # Keep whole chunks already on disk, drop any ragged tail so every byte
    # written here came from a verified full chunk.
    have = target.stat().st_size if target.exists() else 0
    done = (have // CHUNK) * CHUNK
    mode = "r+b" if target.exists() else "wb"
    with open(target, mode) as f:
        f.truncate(done)
        f.seek(done)
        t0 = time.time()
        pos = done
        while pos < TOTAL:
            end = min(pos + CHUNK, TOTAL) - 1
            f.write(fetch_range(pos, end))
            f.flush()
            pos = end + 1
            pct = pos / TOTAL * 100
            rate = (pos - done) / max(time.time() - t0, 1e-9) / 1e6
            eta = (TOTAL - pos) / 1e6 / max(rate, 1e-9) / 60
            print(f"  {pos / 1e6:7.1f}/{TOTAL / 1e6:.0f} MB ({pct:5.1f}%) "
                  f"{rate:.2f} MB/s eta {eta:.1f}m", flush=True)

    print("verifying sha256 ...", flush=True)
    h = hashlib.sha256()
    with open(target, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != SHA256:
        print(f"CHECKSUM MISMATCH\n  got      {h.hexdigest()}\n  expected {SHA256}")
        print("removing the corrupt file, rerun to download cleanly")
        os.remove(target)
        return 1
    print(f"OK: {target} verified ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
