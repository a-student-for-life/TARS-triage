import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import time

import asf_search as asf
import httpx

import config
from models import ImageAlignmentStatus, ImageStatus

ALIGNED_HOURS = 12
WEAK_ALIGNMENT_HOURS = 48
BACKEND_PUBLIC_BASE_URL = os.getenv("BACKEND_PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
GRANULE_TIMESTAMP_RE = re.compile(r"(20\d{6}T\d{6})")


def ensure_utc(value):
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            value = datetime.fromisoformat(normalized)
        except ValueError:
            try:
                value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def classify_alignment(anomaly_timestamp, image_timestamp):
    anomaly_utc = ensure_utc(anomaly_timestamp)
    image_utc = ensure_utc(image_timestamp)
    if anomaly_utc is None or image_utc is None:
        return None, None

    delta_hours = (image_utc - anomaly_utc).total_seconds() / 3600
    if delta_hours < -config.IMAGE_MAX_BEFORE_HOURS:
        return ImageAlignmentStatus.MISALIGNED, delta_hours
    if delta_hours <= ALIGNED_HOURS:
        return ImageAlignmentStatus.ALIGNED, delta_hours
    if delta_hours <= WEAK_ALIGNMENT_HOURS:
        return ImageAlignmentStatus.WEAK_ALIGNMENT, delta_hours
    return ImageAlignmentStatus.MISALIGNED, delta_hours


def _extract_timestamp_from_granule(granule_id):
    if not granule_id:
        return None

    match = GRANULE_TIMESTAMP_RE.search(granule_id)
    if not match:
        return None

    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _extract_image_timestamp(product):
    start_time = ensure_utc(product.properties.get("startTime"))
    if start_time is not None:
        return start_time

    stop_time = ensure_utc(product.properties.get("stopTime"))
    if stop_time is not None:
        return stop_time

    return _extract_timestamp_from_granule(product.properties.get("sceneName"))


def _build_candidate(product, anomaly_timestamp):
    granule_id = product.properties.get("sceneName")
    image_timestamp = _extract_image_timestamp(product)
    alignment_status, delta_hours = classify_alignment(anomaly_timestamp, image_timestamp)
    browse_urls = product.properties.get("browse") or []
    browse_url = browse_urls[0] if browse_urls else None

    return {
        "granule_id": granule_id,
        "browse_url": browse_url,
        "image_timestamp": image_timestamp,
        "image_alignment_status": alignment_status,
        "delta_hours": delta_hours,
        "image_status": ImageStatus.AVAILABLE if browse_url else ImageStatus.UNAVAILABLE,
        "footprint": getattr(product, "geometry", None),  # GeoJSON polygon, used for cropping
    }


def _search_candidates(lat, lon, start_time, end_time, max_results=10):
    roi = f"POINT({lon} {lat})"
    return asf.search(
        platform="Sentinel-1",
        intersectsWith=roi,
        start=start_time,
        end=end_time,
        processingLevel="GRD_HD",
        maxResults=max_results,
    )


def choose_candidate(candidates, anomaly_timestamp, exclude=()):
    """Pick the image to classify for an anomaly.

    Only images acquired after the anomaly (minus a small clock tolerance) and
    within IMAGE_MAX_AFTER_HOURS are eligible: an image taken before the event
    cannot show its consequences. Among eligible images the earliest one with a
    browse preview wins, since it is the fastest possible confirmation.
    """
    anomaly_utc = ensure_utc(anomaly_timestamp)
    if anomaly_utc is None:
        return None
    earliest = anomaly_utc - timedelta(hours=config.IMAGE_MAX_BEFORE_HOURS)
    latest = anomaly_utc + timedelta(hours=config.IMAGE_MAX_AFTER_HOURS)

    eligible = [
        c for c in candidates
        if c["image_timestamp"] is not None and earliest <= c["image_timestamp"] <= latest
        and c["granule_id"] not in exclude
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda c: (0 if c["browse_url"] else 1, c["image_timestamp"]))


def find_relevant_image(lat: float, lon: float, anomaly_timestamp, exclude=()):
    """
    Searches for the first Sentinel-1 GRD preview acquired after the anomaly.
    Returns None if none exists yet; the watchdog retries with backoff.
    """
    anomaly_utc = ensure_utc(anomaly_timestamp) or datetime.now(timezone.utc)
    start = anomaly_utc - timedelta(hours=config.IMAGE_MAX_BEFORE_HOURS)
    end = anomaly_utc + timedelta(hours=config.IMAGE_MAX_AFTER_HOURS)

    print(
        f"   [ASF] Searching {start.isoformat()} .. {end.isoformat()} for {lat}, {lon}...",
        flush=True,
    )
    try:
        results = _search_candidates(lat, lon, start, end)
    except Exception as exc:
        print(f"   [ASF] API Error: {exc}", flush=True)
        raise

    candidates = [_build_candidate(product, anomaly_utc) for product in results]
    best = choose_candidate(candidates, anomaly_utc, exclude)
    if best is None:
        print("   [ASF] No image acquired after the anomaly yet.", flush=True)
    else:
        print(f"   [ASF] Selected {best['granule_id']} ({best['image_alignment_status']})", flush=True)
    return best


async def _download_preview_async(image_url, save_path):
    timeout = httpx.Timeout(15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", image_url) as response:
            if response.status_code != 200:
                print(f"   [ASF] HTTP Error: {response.status_code}", flush=True)
                return None

            content_type = (response.headers.get("Content-Type") or "").lower()
            if content_type and "image" not in content_type:
                print(f"   [ASF] Unexpected content type: {content_type}", flush=True)
                return None

            bytes_written = 0
            with open(save_path, "wb") as file_handle:
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if not chunk:
                        continue
                    file_handle.write(chunk)
                    bytes_written += len(chunk)

            if bytes_written == 0:
                print("   [ASF] Downloaded preview was empty.", flush=True)
                return None

            print(f"   [ASF] Saved to: {save_path}", flush=True)
            return save_path


def download_preview(image_url, save_dir=config.TEMP_IMAGE_DIR):
    """
    Downloads the JPG preview from the URL with retry logic.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    filename = image_url.split("/")[-1].split("?")[0]
    save_path = os.path.join(save_dir, filename)

    print(f"   [ASF] Downloading preview: {filename}...", flush=True)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            return asyncio.run(_download_preview_async(image_url, save_path))
        except Exception as exc:
            print(
                f"   [ASF] Download Error (Attempt {attempt + 1}/{max_retries}): {exc}",
                flush=True,
            )
            time.sleep(2)

    print(f"   [ASF] Failed to download after {max_retries} attempts.", flush=True)
    return None


def build_public_image_url(local_path):
    return f"{BACKEND_PUBLIC_BASE_URL}/images/{Path(local_path).name}"


def find_latest_image(lat: float, lon: float):
    """
    Backward-compatible wrapper.
    """
    return find_relevant_image(lat, lon, datetime.now(timezone.utc))
