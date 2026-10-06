"""RapidAPI video-download ladder — the SAME integration the live Sapient
pipeline uses for links (ported from RRG/SIMPLR/sapient-model/modal_pipeline/
app.py). Social platforms (IG/TikTok/YT/X/FB) serve HTML, not media, and yt-dlp
gets IP-blocked on datacenter hosts — so RapidAPI is the primary path and yt-dlp
the fallback.

  Tier 0A — Auto Download All In One (auto-download-all-in-one.p.rapidapi.com) — generic
  Tier 0B — Social Media Video Downloader (Instagram fallback)
  Tier 0C — YTStream (YouTube fallback)
  Tier 1  — yt-dlp

All tiers share RAPIDAPI_KEY (the `sapient-rapidapi` Modal secret). Downloads are
ffprobe-validated to reject HTML/captcha error bodies that sneak through HTTP 200.
`download_via_ladder` returns the downloaded media Path or raises ValueError.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

MIN_BYTES = 50_000
# NOTE: m4a (audio-only) intentionally excluded — accepting it let audio-only
# downloads pass the "is this video?" check, so a reel could reach Mary with NO
# frames (the visual/SlowFast stream is active in production and needs them).
VIDEO_FORMATS = ("mov", "mp4", "matroska", "webm")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


def _platform_context(url: str) -> tuple[str, str | None]:
    host = (urlparse(url).netloc or "").lower()
    if "tiktok" in host: return "tiktok", "https://www.tiktok.com"
    if "instagram" in host: return "instagram", "https://www.instagram.com"
    if "youtube" in host or "youtu.be" in host: return "youtube", "https://www.youtube.com"
    if "twitter" in host or "x.com" in host: return "twitter", "https://twitter.com"
    if "facebook" in host or "fb.watch" in host: return "facebook", "https://www.facebook.com"
    return "other", None


def _ffprobe_is_video(path: Path) -> bool:
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
    except Exception:
        return False
    return any(f in (p.stdout or "").strip().lower() for f in VIDEO_FORMATS)


def _save_if_video(content: bytes, out_path: Path) -> bool:
    if len(content) <= MIN_BYTES:
        return False
    out_path.write_bytes(content)
    return _ffprobe_is_video(out_path)


def _tier_auto(url: str, key: str, referer_origin: str | None, out_path: Path) -> Path | None:
    import requests
    try:
        resp = requests.post(
            "https://auto-download-all-in-one.p.rapidapi.com/v1/social/autolink",
            json={"url": url},
            headers={"x-rapidapi-key": key, "x-rapidapi-host": "auto-download-all-in-one.p.rapidapi.com",
                     "Content-Type": "application/json"},
            timeout=45,
        )
        if resp.status_code != 200:
            return None
        medias = (resp.json() or {}).get("medias") or []
    except Exception:
        return None

    candidates: list[str] = [m["url"] for m in medias if m.get("url") and m.get("type") == "video"]
    candidates += [m["url"] for m in medias if m.get("url") and m["url"] not in candidates]
    if not candidates:
        return None

    base = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}
    attempts = ([{**base, "Referer": f"{referer_origin}/", "Origin": referer_origin}, base]
                if referer_origin else [base])
    for vurl in candidates[:5]:
        if not (isinstance(vurl, str) and vurl.startswith("http")):
            continue
        for headers in attempts:
            try:
                r = requests.get(vurl, timeout=90, allow_redirects=True, headers=headers)
            except Exception:
                continue
            if r.status_code == 200 and _save_if_video(r.content, out_path):
                return out_path
    return None


def _tier_smvd(url: str, key: str, out_path: Path) -> Path | None:
    import requests
    try:
        resp = requests.get(
            "https://social-media-video-downloader.p.rapidapi.com/smvd/get/all",
            params={"url": url},
            headers={"x-rapidapi-key": key, "x-rapidapi-host": "social-media-video-downloader.p.rapidapi.com"},
            timeout=45,
        )
        if resp.status_code != 200:
            return None
        links = (resp.json() or {}).get("links") or []
    except Exception:
        return None
    vurl = None
    for link in links:
        if link.get("link") and link.get("quality", "") in ("hd", "sd", "high", "low", ""):
            vurl = link["link"]; break
    if not vurl and links:
        vurl = links[0].get("link")
    if not (vurl and isinstance(vurl, str) and vurl.startswith("http")):
        return None
    try:
        r = requests.get(vurl, timeout=90, allow_redirects=True, headers={"User-Agent": UA})
    except Exception:
        return None
    return out_path if (r.status_code == 200 and _save_if_video(r.content, out_path)) else None


def _tier_ytstream(url: str, key: str, out_path: Path) -> Path | None:
    import requests
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url) or re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    vid = m.group(1) if m else None
    if not vid:
        return None
    try:
        resp = requests.get(
            "https://ytstream-download-youtube-videos.p.rapidapi.com/dl",
            params={"id": vid},
            headers={"x-rapidapi-key": key, "x-rapidapi-host": "ytstream-download-youtube-videos.p.rapidapi.com"},
            timeout=45,
        )
        if resp.status_code != 200:
            return None
        data = resp.json() or {}
    except Exception:
        return None
    vurl = None
    for f in (data.get("formats") or []):
        if f.get("url") and str(f.get("mimeType", "")).startswith("video/"):
            vurl = f["url"]; break
    if not vurl:
        lf = data.get("link") or data.get("links") or []
        if isinstance(lf, list):
            for lnk in lf:
                if isinstance(lnk, dict) and lnk.get("url"):
                    vurl = lnk["url"]; break
        elif isinstance(lf, str) and lf.startswith("http"):
            vurl = lf
    if not (vurl and isinstance(vurl, str) and vurl.startswith("http")):
        return None
    try:
        r = requests.get(vurl, timeout=120, allow_redirects=True, headers={"User-Agent": UA})
    except Exception:
        return None
    return out_path if (r.status_code == 200 and _save_if_video(r.content, out_path)) else None


def _normalize_url(url: str) -> str:
    """Make a pasted link well-formed. Users often paste a bare host
    (`www.instagram.com/reel/…`) with no scheme; without one urlparse puts
    everything in `path` and downloaders fail. Add https:// and strip junk."""
    u = (url or "").strip().strip('"').strip("'").strip()
    if u.startswith("//"):
        return "https:" + u
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        return "https://" + u.lstrip("/")
    return u


def download_via_ladder(source_url: str, out_dir: str) -> str:
    """RapidAPI tiers (0A→0B→0C) → yt-dlp fallback. Returns a media file path or
    raises ValueError with a clear message if every tier fails."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    source_url = _normalize_url(source_url)
    platform, referer = _platform_context(source_url)
    key = os.environ.get("RAPIDAPI_KEY", "")
    out = d / "video.mp4"

    # Tier -1 — DIRECT media URL (ends in a video extension). For a plain http(s)
    # link to an .mp4/.mov/.webm, a browser-UA GET is the most reliable path; the
    # RapidAPI social tiers mangle direct links and yt-dlp's generic extractor often
    # gets IP-blocked (403/526) on direct hosts. ffprobe-validated like every tier.
    pl = urlparse(source_url).path.lower()
    if pl.endswith((".mp4", ".mov", ".webm", ".mkv", ".m4v")):
        import requests
        for hdr in ({"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"},
                    {"User-Agent": "curl/8.0"}):
            try:
                r = requests.get(source_url, timeout=120, allow_redirects=True, headers=hdr)
                if r.status_code == 200 and _save_if_video(r.content, out):
                    return str(out)
            except Exception:
                continue

    if key:
        r = _tier_auto(source_url, key, referer, out)
        if r is not None:
            return str(r)
        if platform == "instagram":
            r = _tier_smvd(source_url, key, out)
            if r is not None:
                return str(r)
        if platform == "youtube":
            r = _tier_ytstream(source_url, key, out)
            if r is not None:
                return str(r)

    # Fallback — yt-dlp. Prefer a real video+audio file (single-file mp4 first so
    # no ffmpeg merge is needed, then video+audio merge, then anything). The old
    # "bestaudio/best" grabbed audio-only, starving Mary's visual encoder of frames.
    tmpl = str(d / "video.%(ext)s")
    proc = subprocess.run(
        ["yt-dlp", "-q", "--no-warnings", "--no-playlist", "-f", "b[ext=mp4]/bv*+ba/b",
         "--user-agent", UA, "-o", tmpl, source_url],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    import glob
    files = [f for f in glob.glob(str(d / "video.*"))]
    if files:
        return files[0]
    raise ValueError(
        f"couldn't fetch media from that {platform if platform != 'other' else 'link'} "
        f"(it may be private or login-required). {(proc.stderr or '').strip()[:160]}"
    )
