import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from modules.utils import parse_timestamp


def count_per_frame(tracking_results: list[dict]) -> list[dict]:
    """Extract pig count per frame from tracking results."""
    return [
        {
            "frame_idx": r["frame_idx"],
            "frame_path": r["frame_path"],
            "count": len(r["tracks"]),
        }
        for r in tracking_results
    ]


def compute_stats(counts: list[dict]) -> dict:
    values = [c["count"] for c in counts]
    return {
        "mean": float(np.mean(values)),
        "max": int(np.max(values)),
        "min": int(np.min(values)),
        "total_frames": len(values),
    }


def plot_count(counts: list[dict], save_path: str = None) -> plt.Figure:
    """Plot pig count over frame index."""
    frames = [c["frame_idx"] for c in counts]
    values = [c["count"] for c in counts]
    stats = compute_stats(counts)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frames, values, marker="o", color="#e07b39", linewidth=2, markersize=4)
    ax.axhline(stats["mean"], color="gray", linestyle="--", linewidth=1, label=f"Mean: {stats['mean']:.1f}")
    ax.fill_between(frames, values, alpha=0.15, color="#e07b39")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pig Count")
    ax.set_title("Pig Count per Frame (all sessions)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_count_timeline(tracking_results: list[dict], date_filter: str = None, save_path: str = None) -> plt.Figure:
    """
    Plot pig count on a real time axis using filename timestamps.
    date_filter: e.g. '2022-01-09' to show only that day.
    """
    entries = []
    for r in tracking_results:
        ts = parse_timestamp(r["frame_path"])
        if ts is None:
            continue
        if date_filter and ts.strftime("%Y-%m-%d") != date_filter:
            continue
        entries.append((ts, len(r["tracks"])))

    if not entries:
        return None

    entries.sort(key=lambda x: x[0])
    times, values = zip(*entries)
    mean_val = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, values, marker="o", color="#4c8be0", linewidth=2, markersize=4)
    ax.axhline(mean_val, color="gray", linestyle="--", linewidth=1, label=f"Mean: {mean_val:.1f}")
    ax.fill_between(times, values, alpha=0.15, color="#4c8be0")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    ax.set_xlabel("Time")
    ax.set_ylabel("Pig Count")
    title = f"Pig Count over Time — {date_filter}" if date_filter else "Pig Count over Time"
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
