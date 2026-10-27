"""Pull live public posts from real social platforms.

Reddit now returns 403 to anonymous JSON, Bluesky's public AppView 403s too, and
X is paid, so I use the two that still serve public timelines without a key:
Mastodon instances and Lemmy. Both are real platforms with real users, which is
the point. Nothing here needs credentials.
"""

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

UA = "nlp-coursework-hate-speech-detection/0.1 (academic project)"
TAG_RE = re.compile(r"<[^>]+>")
BLOCK_RE = re.compile(r"</?(?:p|br|div|li|ul|ol|blockquote)\b[^>]*>", re.IGNORECASE)

# Instances that answered an anonymous public-timeline request. mastodon.social
# and mstdn.social both require auth now, so they are deliberately absent.
MASTODON_INSTANCES = [
    "mastodon.world",
    "fosstodon.org",
    "techhub.social",
    "mas.to",
]

LEMMY_INSTANCES = ["lemmy.world"]


def _get(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _strip(markup: str) -> str:
    """Mastodon serves HTML. Keep the text, keep the emoji, drop the tags.

    Block tags become a space and inline tags vanish. Replacing every tag with a
    space split hashtags and URLs apart, turning `#BBC` into `# BBC`, which is
    exactly the markup the classifier should never see.
    """
    text = BLOCK_RE.sub(" ", markup or "")
    text = TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def from_mastodon(instance: str, limit: int = 40, local: bool = False) -> list:
    q = urllib.parse.urlencode({"limit": min(limit, 40), "local": str(local).lower()})
    try:
        posts = _get(f"https://{instance}/api/v1/timelines/public?{q}")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as e:
        return [{"_error": f"{instance}: {e}"}]
    out = []
    for p in posts:
        text = _strip(p.get("content", ""))
        if not text:
            continue
        acct = (p.get("account") or {}).get("acct", "unknown")
        out.append({
            "text": text,
            "author": f"@{acct}",
            "platform": f"Mastodon ({instance})",
            "url": p.get("url") or "",
            "created_at": p.get("created_at", ""),
            "declared_lang": p.get("language") or "",
        })
    return out


def from_mastodon_tag(instance: str, tag: str, limit: int = 40) -> list:
    """Hashtag timeline. Public timelines skew benign, so topical pulls give the
    moderation queue something to actually triage."""
    tag = tag.lstrip("#").strip()
    if not tag:
        return []
    q = urllib.parse.urlencode({"limit": min(limit, 40)})
    try:
        posts = _get(f"https://{instance}/api/v1/timelines/tag/{urllib.parse.quote(tag)}?{q}")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as e:
        return [{"_error": f"{instance} #{tag}: {e}"}]
    out = []
    for p in posts:
        text = _strip(p.get("content", ""))
        if not text:
            continue
        out.append({
            "text": text,
            "author": f"@{(p.get('account') or {}).get('acct', 'unknown')}",
            "platform": f"Mastodon #{tag}",
            "url": p.get("url") or "",
            "created_at": p.get("created_at", ""),
            "declared_lang": p.get("language") or "",
        })
    return out


def from_lemmy(instance: str = "lemmy.world", limit: int = 40, sort: str = "New") -> list:
    q = urllib.parse.urlencode({"limit": min(limit, 50), "sort": sort, "type_": "All"})
    try:
        data = _get(f"https://{instance}/api/v3/post/list?{q}")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as e:
        return [{"_error": f"{instance}: {e}"}]
    out = []
    for item in data.get("posts", []):
        post = item.get("post", {})
        # Title plus body, since a Lemmy post often carries its point in the title.
        text = " ".join(x for x in (post.get("name"), post.get("body")) if x).strip()
        if not text:
            continue
        creator = (item.get("creator") or {}).get("name", "unknown")
        community = (item.get("community") or {}).get("name", "")
        out.append({
            "text": text[:2000],
            "author": f"u/{creator}",
            "platform": f"Lemmy (c/{community})",
            "url": post.get("ap_id") or "",
            "created_at": post.get("published", ""),
            "declared_lang": "",
        })
    return out


def fetch(sources: list, per_source: int = 40, tag: str = "") -> tuple:
    """Collect posts from the chosen sources. Returns (posts, errors).

    A tag switches the Mastodon sources to their hashtag timelines. Lemmy has no
    equivalent open endpoint, so it keeps serving its newest posts.
    """
    posts, errors, seen = [], [], set()
    for src in sources:
        if src.startswith("lemmy:"):
            rows = from_lemmy(src.split("lemmy:", 1)[1], limit=per_source)
        elif tag:
            rows = from_mastodon_tag(src, tag, limit=per_source)
        else:
            rows = from_mastodon(src, limit=per_source)
        for r in rows:
            if "_error" in r:
                errors.append(r["_error"])
                continue
            key = r["text"][:120]
            if key in seen:
                continue
            seen.add(key)
            posts.append(r)
    return posts, errors


def age_minutes(iso: str) -> float:
    try:
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).total_seconds() / 60
    except Exception:
        return float("nan")
