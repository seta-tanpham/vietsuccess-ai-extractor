"""
YouTube crawl source (yt-dlp wrapper).

No OAuth / API key / quota. Fetches metadata and downloads a merged mp4 so the
existing pipeline (Phase 1 normalize → STT → diarization → ...) can run unchanged.

Public API:
    extract_video_id(url)            -> "dQw4w9WgXcQ"
    canonicalize_url(url)            -> "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    fetch_metadata(url)              -> dict (full yt-dlp info, download=False)
    download_media(url, dest_dir)    -> (local_path, info_dict)
    map_metadata(info)               -> dict of normalized fields for the Video/Channel rows
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import settings

log = logging.getLogger(__name__)


class YouTubeCrawlError(RuntimeError):
    """Raised when a URL is not a usable YouTube video or the crawl fails."""


_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def extract_video_id(url: str) -> str:
    """Pull the 11-char YouTube video id out of any common URL form."""
    # Shells (zsh) often backslash-escape ?, =, & when a URL is pasted unquoted.
    url = url.replace("\\", "").strip()
    patterns = [
        r"(?:v=|/shorts/|/embed/|/v/)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    # Bare id passed directly
    stripped = url.strip()
    if _ID_RE.fullmatch(stripped):
        return stripped
    raise YouTubeCrawlError(f"Could not extract a YouTube video id from: {url!r}")


def canonicalize_url(url: str) -> str:
    """Normalize to the canonical watch URL so the same video dedups regardless of
    tracking params, shorts/embed forms, or youtu.be shortlinks."""
    return f"https://www.youtube.com/watch?v={extract_video_id(url)}"


def _ydl_opts(dest_dir: Optional[str] = None, download: bool = False) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": not download,
        # Anti-bot: space out metadata requests a little.
        "sleep_interval_requests": settings.youtube_sleep_requests_s,
    }
    if download:
        opts.update({
            "format": settings.youtube_format,
            "merge_output_format": "mp4",
            "outtmpl": str(Path(dest_dir) / "%(id)s.%(ext)s"),
            # Anti-bot: random pause in [sleep_interval, max_sleep_interval] before download.
            "sleep_interval": settings.youtube_sleep_interval_s,
            "max_sleep_interval": settings.youtube_max_sleep_interval_s,
        })
    return opts


def extract_playlist_entries(url: str) -> dict[str, Any]:
    """List a playlist's videos WITHOUT downloading (flat extraction).

    Returns {"title": str, "videos": [{"id", "title", "url"}]}.
    """
    import yt_dlp

    url = url.replace("\\", "").strip()
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,   # don't resolve each video, just list them
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise YouTubeCrawlError(f"Failed to list playlist {url!r}: {exc}") from exc

    entries = info.get("entries") or []
    videos = []
    for e in entries:
        if not e or not e.get("id"):
            continue
        videos.append({
            "id": e["id"],
            "title": e.get("title"),
            "url": e.get("url") or f"https://www.youtube.com/watch?v={e['id']}",
        })
    return {"title": info.get("title"), "videos": videos}


def fetch_metadata(url: str) -> dict[str, Any]:
    """Return the full yt-dlp info dict without downloading media."""
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL(_ydl_opts(download=False)) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt_dlp.utils.DownloadError and friends
        raise YouTubeCrawlError(f"Failed to fetch metadata for {url!r}: {exc}") from exc
    if not info:
        raise YouTubeCrawlError(f"No metadata returned for {url!r}")
    return info


def download_media(url: str, dest_dir: Optional[str] = None) -> tuple[Path, dict[str, Any]]:
    """Download merged mp4 to dest_dir. Returns (local_path, info_dict)."""
    import yt_dlp

    dest = Path(dest_dir or settings.youtube_download_dir)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        with yt_dlp.YoutubeDL(_ydl_opts(str(dest), download=True)) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as exc:
        raise YouTubeCrawlError(f"Failed to download {url!r}: {exc}") from exc

    # merge_output_format may have rewritten the extension to .mp4
    if not path.exists():
        mp4 = path.with_suffix(".mp4")
        if mp4.exists():
            path = mp4
    if not path.exists():
        raise YouTubeCrawlError(f"Downloaded file not found for {url!r} (expected {path})")

    log.info("[crawl] downloaded %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return path, info


# ── Metadata mapping ──────────────────────────────────────────────────────────

def _parse_upload_date(info: dict) -> Optional[datetime]:
    # yt-dlp exposes `upload_date` as "YYYYMMDD" and sometimes `timestamp` (epoch).
    ts = info.get("timestamp")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    raw = info.get("upload_date")
    if raw and len(str(raw)) == 8:
        try:
            return datetime.strptime(str(raw), "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _best_thumbnail(info: dict) -> Optional[str]:
    thumbs = info.get("thumbnails") or []
    if thumbs:
        # yt-dlp lists thumbnails worst → best; last is highest res.
        return thumbs[-1].get("url")
    return info.get("thumbnail")


def map_metadata(info: dict) -> dict[str, Any]:
    """Normalize a yt-dlp info dict into fields used by the Video + Channel rows."""
    duration = info.get("duration")  # seconds
    return {
        "youtube_video_id": info.get("id"),
        "title": info.get("title"),
        "description": info.get("description"),
        "duration_ms": int(duration * 1000) if isinstance(duration, (int, float)) else None,
        "published_at": _parse_upload_date(info),
        "tags": info.get("tags") or None,
        "youtube_category_id": (info.get("categories") or [None])[0],
        "default_language": info.get("language"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "thumbnail_url": _best_thumbnail(info),
        "channel": {
            "youtube_channel_id": info.get("channel_id"),
            "title": info.get("channel") or info.get("uploader"),
            "custom_url": info.get("uploader_id"),
            "subscriber_count": info.get("channel_follower_count"),
            "thumbnail_url": None,
        },
    }
