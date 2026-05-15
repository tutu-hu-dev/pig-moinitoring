import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.utils import parse_timestamp


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def compute_crowding(frame_result: dict, frame_w: int, frame_h: int) -> dict:
    """
    Compute crowding metrics for a single frame.

    Returns:
      count        - number of pigs detected
      density      - fraction of frame area covered by bounding boxes
      overlap_rate - fraction of pig pairs that overlap
      score        - combined crowding score in [0, 1]
      alert        - True if score exceeds threshold
    """
    tracks = frame_result["tracks"]
    count = len(tracks)
    frame_area = frame_w * frame_h

    if count == 0:
        return {"count": 0, "density": 0.0, "overlap_rate": 0.0, "score": 0.0, "alert": False}

    boxes = [t["box"] for t in tracks]

    # density: total box area / frame area (capped at 1)
    total_box_area = sum(_box_area(b) for b in boxes)
    density = min(total_box_area / frame_area, 1.0)

    # overlap_rate: fraction of pairs that have any overlap
    n_pairs = count * (count - 1) / 2
    if n_pairs > 0:
        overlapping = sum(
            1 for i in range(count) for j in range(i + 1, count)
            if _intersection_area(boxes[i], boxes[j]) > 0
        )
        overlap_rate = overlapping / n_pairs
    else:
        overlap_rate = 0.0

    # combined score: weight density and overlap equally
    score = 0.5 * density + 0.5 * overlap_rate

    return {
        "count": count,
        "density": density,
        "overlap_rate": overlap_rate,
        "score": score,
        "alert": score > 0.3,
    }


def compute_crowding_all(tracking_results: list[dict], ref_image_path: str) -> list[dict]:
    """Run crowding analysis on all frames."""
    ref = cv2.imread(ref_image_path)
    h, w = ref.shape[:2]
    results = []
    for r in tracking_results:
        metrics = compute_crowding(r, w, h)
        metrics["frame_idx"] = r["frame_idx"]
        metrics["frame_path"] = r["frame_path"]
        results.append(metrics)
    return results


def plot_crowding(crowding_results: list[dict], save_path: str = None) -> plt.Figure:
    """Plot crowding score and pig count over frames, highlight alert frames."""
    frames = [c["frame_idx"] for c in crowding_results]
    scores = [c["score"] for c in crowding_results]
    counts = [c["count"] for c in crowding_results]
    alerts = [c["alert"] for c in crowding_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # crowding score
    ax1.plot(frames, scores, color="#c0392b", linewidth=2)
    ax1.axhline(0.3, color="orange", linestyle="--", linewidth=1, label="Alert threshold (0.3)")
    ax1.fill_between(frames, scores, 0.3,
                     where=[s > 0.3 for s in scores],
                     color="#e74c3c", alpha=0.3, label="Crowded")
    ax1.set_ylabel("Crowding Score")
    ax1.set_title("Crowding Analysis")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # pig count with alert markers
    ax2.plot(frames, counts, color="#2980b9", linewidth=2)
    alert_frames = [f for f, a in zip(frames, alerts) if a]
    alert_counts = [c for c, a in zip(counts, alerts) if a]
    ax2.scatter(alert_frames, alert_counts, color="#e74c3c", zorder=5, s=40, label="Alert frame")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Pig Count")
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_crowding_timeline(crowding_results: list[dict], date_filter: str = None, save_path: str = None) -> plt.Figure:
    """Plot crowding score on a real timestamp axis."""
    import matplotlib.dates as mdates

    entries = []
    for c in crowding_results:
        ts = parse_timestamp(c["frame_path"])
        if ts is None:
            continue
        if date_filter and ts.strftime("%Y-%m-%d") != date_filter:
            continue
        entries.append((ts, c["score"], c["alert"]))

    if not entries:
        return None

    entries.sort(key=lambda x: x[0])
    times, scores, alerts = zip(*entries)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, scores, color="#c0392b", linewidth=2)
    ax.axhline(0.3, color="orange", linestyle="--", linewidth=1, label="Alert threshold")
    ax.fill_between(times, scores, 0.3,
                    where=[s > 0.3 for s in scores],
                    color="#e74c3c", alpha=0.3, label="Crowded")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    title = f"Crowding Score over Time — {date_filter}" if date_filter else "Crowding Score over Time"
    ax.set_title(title)
    ax.set_ylabel("Crowding Score")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
