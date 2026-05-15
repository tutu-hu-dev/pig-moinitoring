from pathlib import Path
from modules.tracker import run_tracking, save_tracked_video
from modules.counter import count_per_frame, compute_stats, plot_count, plot_count_timeline
from modules.heatmap import build_heatmap, plot_heatmap, plot_grouped_heatmaps
from modules.crowding import compute_crowding_all, plot_crowding, plot_crowding_timeline
from modules.utils import group_by_date, group_by_period

BASE = Path(__file__).parent

FRAMES_DIR = r"C:\Users\zhouh\OneDrive\Documents\Notes\tutubox\machine-learning-note\PigDetection-1\train\images"
MODEL_PATH = str(BASE / "models" / "best.pt")
OUTPUT_VIDEO      = str(BASE / "output_tracked.mp4")
OUTPUT_CHART      = str(BASE / "output_count.png")
OUTPUT_TIMELINE   = str(BASE / "output_count_jan9.png")
OUTPUT_HEATMAP    = str(BASE / "output_heatmap.png")
OUTPUT_HM_DATE    = str(BASE / "output_heatmap_by_date.png")
OUTPUT_HM_PERIOD  = str(BASE / "output_heatmap_morning_vs_evening.png")
OUTPUT_CROWDING   = str(BASE / "output_crowding.png")
OUTPUT_CROWD_JAN9 = str(BASE / "output_crowding_jan9.png")

if __name__ == "__main__":
    # --- tracking ---
    print("Running tracking...")
    results = run_tracking(FRAMES_DIR, MODEL_PATH, conf=0.3)
    for r in results:
        ids = [t["id"] for t in r["tracks"]]
        print(f"Frame {r['frame_idx']:>3}: {len(r['tracks'])} pigs  IDs={ids}")
    save_tracked_video(results, OUTPUT_VIDEO, fps=3.0)
    print(f"Saved tracked video → {OUTPUT_VIDEO}")

    # --- counting ---
    print("\nRunning counter...")
    counts = count_per_frame(results)
    stats = compute_stats(counts)
    print(f"Mean: {stats['mean']:.1f}  Max: {stats['max']}  Min: {stats['min']}")
    plot_count(counts, save_path=OUTPUT_CHART)
    print(f"Saved count chart → {OUTPUT_CHART}")

    # Jan 9 timeline (most continuous session, ~170 frames over 11 min)
    plot_count_timeline(results, date_filter="2022-01-09", save_path=OUTPUT_TIMELINE)
    print(f"Saved Jan-9 timeline → {OUTPUT_TIMELINE}")

    # --- heatmaps ---
    ref_path = results[0]["frame_path"]
    print("\nBuilding heatmaps...")

    heatmap = build_heatmap(results, ref_path)
    plot_heatmap(heatmap, ref_path, save_path=OUTPUT_HEATMAP)
    print(f"Saved overall heatmap → {OUTPUT_HEATMAP}")

    by_date = group_by_date(results)
    plot_grouped_heatmaps(by_date, ref_path, save_path=OUTPUT_HM_DATE)
    print(f"Saved per-date heatmap → {OUTPUT_HM_DATE}")

    by_period = group_by_period(results)
    plot_grouped_heatmaps(by_period, ref_path, title_prefix="", save_path=OUTPUT_HM_PERIOD)
    print(f"Saved morning vs evening heatmap → {OUTPUT_HM_PERIOD}")

    # --- crowding ---
    print("\nRunning crowding analysis...")
    crowding = compute_crowding_all(results, ref_path)
    alert_count = sum(1 for c in crowding if c["alert"])
    print(f"Alert frames: {alert_count} / {len(crowding)} ({100*alert_count/len(crowding):.1f}%)")
    plot_crowding(crowding, save_path=OUTPUT_CROWDING)
    print(f"Saved crowding chart → {OUTPUT_CROWDING}")
    plot_crowding_timeline(crowding, date_filter="2022-01-09", save_path=OUTPUT_CROWD_JAN9)
    print(f"Saved Jan-9 crowding timeline → {OUTPUT_CROWD_JAN9}")

    print("\nDone.")
