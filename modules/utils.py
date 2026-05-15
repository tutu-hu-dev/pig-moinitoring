import re
from datetime import datetime
from pathlib import Path


def parse_timestamp(frame_path: str) -> datetime | None:
    """Extract datetime from vlcsnap filename."""
    name = Path(frame_path).stem
    m = re.search(r"(\d{4}-\d{2}-\d{2})-(\d{2})h(\d{2})m(\d{2})s", name)
    if not m:
        return None
    date_str, h, minute, s = m.group(1), m.group(2), m.group(3), m.group(4)
    return datetime.strptime(f"{date_str} {h}:{minute}:{s}", "%Y-%m-%d %H:%M:%S")


def group_by_date(tracking_results: list[dict]) -> dict[str, list[dict]]:
    """Group frames by date string (e.g. '2022-01-09')."""
    groups: dict[str, list[dict]] = {}
    for r in tracking_results:
        ts = parse_timestamp(r["frame_path"])
        if ts is None:
            continue
        key = ts.strftime("%Y-%m-%d")
        groups.setdefault(key, []).append(r)
    return groups


def group_by_period(tracking_results: list[dict]) -> dict[str, list[dict]]:
    """Group frames into 'morning' (before 14:00) and 'evening' (14:00+)."""
    groups: dict[str, list[dict]] = {"morning": [], "evening": []}
    for r in tracking_results:
        ts = parse_timestamp(r["frame_path"])
        if ts is None:
            continue
        key = "morning" if ts.hour < 14 else "evening"
        groups[key].append(r)
    return groups
