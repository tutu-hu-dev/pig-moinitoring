import tempfile
import shutil
import streamlit as st
from pathlib import Path
from modules.tracker import run_tracking, save_tracked_video
from modules.counter import count_per_frame, compute_stats, plot_count, plot_count_timeline
from modules.heatmap import build_heatmap, plot_heatmap, plot_grouped_heatmaps
from modules.crowding import compute_crowding_all, plot_crowding, plot_crowding_timeline
from modules.utils import group_by_date, group_by_period

BASE        = Path(__file__).parent
MODEL_PATH  = str(BASE / "models" / "best.pt")
SAMPLES_DIR = BASE / "data" / "samples"

st.set_page_config(page_title="Pig Monitoring System", layout="wide")
st.title("Pig Monitoring & Behavior Analysis")
st.caption("YOLOv8 + ByteTrack — detection, tracking, heatmaps, and crowding alerts")

# ── sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    mode = st.radio("Choose input mode", ["Use sample data", "Upload my own images"])

    uploaded_files = None
    if mode == "Upload my own images":
        uploaded_files = st.file_uploader(
            "Upload pig images (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if not uploaded_files:
            st.info("Please upload at least one image.")

    st.divider()
    conf = st.slider("Detection confidence", 0.1, 0.9, 0.3, 0.05)
    run_btn = st.button("Run Analysis", type="primary")


def prepare_frames_dir(mode, uploaded_files) -> str | None:
    """Return a directory path containing the frames to analyse."""
    if mode == "Use sample data":
        return str(SAMPLES_DIR)
    if uploaded_files:
        tmp_dir = Path(tempfile.mkdtemp())
        for f in uploaded_files:
            (tmp_dir / f.name).write_bytes(f.read())
        return str(tmp_dir)
    return None


@st.cache_data(show_spinner="Running YOLO tracking — this may take a minute...")
def run_pipeline(frames_dir: str, model_path: str, conf: float):
    results = run_tracking(frames_dir, model_path, conf=conf)
    video_path = str(BASE / "output_tracked.mp4")
    save_tracked_video(results, video_path, fps=3.0)
    return results, video_path


# ── run ──────────────────────────────────────────────────────────
if run_btn:
    frames_dir = prepare_frames_dir(mode, uploaded_files)
    if frames_dir is None:
        st.warning("Please upload images first.")
    else:
        st.session_state["results"], st.session_state["video_path"] = run_pipeline(
            frames_dir, MODEL_PATH, conf
        )
        st.session_state["mode_label"] = (
            "Sample data" if mode == "Use sample data"
            else f"Uploaded ({len(uploaded_files)} images)"
        )

if "results" not in st.session_state:
    st.info("Select a data source in the sidebar and click **Run Analysis** to start.")
    st.stop()

results    = st.session_state["results"]
video_path = st.session_state["video_path"]
ref_path   = results[0]["frame_path"]

st.caption(f"Source: {st.session_state.get('mode_label', '')}  |  {len(results)} frames processed")

tab1, tab2, tab3, tab4 = st.tabs(["Tracking", "Count", "Heatmap", "Crowding"])

# ── Tab 1: Tracking ──────────────────────────────────────────────
with tab1:
    st.subheader("Tracked Video")
    if Path(video_path).exists():
        st.video(video_path)
    else:
        st.warning("Video not found.")

    st.subheader("Per-frame Summary")
    rows = [
        {
            "Frame": r["frame_idx"],
            "Pig Count": len(r["tracks"]),
            "Track IDs": str([t["id"] for t in r["tracks"]]),
        }
        for r in results
    ]
    st.dataframe(rows, use_container_width=True)

# ── Tab 2: Count ─────────────────────────────────────────────────
with tab2:
    counts = count_per_frame(results)
    stats  = compute_stats(counts)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Frames", stats["total_frames"])
    c2.metric("Mean Count",   f"{stats['mean']:.1f}")
    c3.metric("Max Count",    stats["max"])
    c4.metric("Min Count",    stats["min"])

    st.subheader("Count over Frames")
    st.pyplot(plot_count(counts))

    st.subheader("Count over Time — Jan 9 (continuous 11-min session)")
    fig = plot_count_timeline(results, date_filter="2022-01-09")
    if fig:
        st.pyplot(fig)
    else:
        st.info("No Jan-9 timestamp frames in this dataset (only available with sample data).")

# ── Tab 3: Heatmap ───────────────────────────────────────────────
with tab3:
    st.subheader("Overall Activity Heatmap")
    heatmap = build_heatmap(results, ref_path)
    st.pyplot(plot_heatmap(heatmap, ref_path))

    by_date = group_by_date(results)
    if len(by_date) > 1:
        st.subheader("Heatmap by Date")
        st.pyplot(plot_grouped_heatmaps(by_date, ref_path))

        by_period = group_by_period(results)
        if by_period["morning"] and by_period["evening"]:
            st.subheader("Morning vs Evening")
            st.pyplot(plot_grouped_heatmaps(by_period, ref_path))
    else:
        st.info("Date comparison requires images from multiple days (available with sample data).")

# ── Tab 4: Crowding ──────────────────────────────────────────────
with tab4:
    crowding = compute_crowding_all(results, ref_path)
    alert_n  = sum(1 for c in crowding if c["alert"])
    total_n  = len(crowding)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Frames", total_n)
    c2.metric("Alert Frames", alert_n)
    c3.metric("Alert Rate",   f"{100 * alert_n / total_n:.1f}%")

    st.subheader("Crowding Score over All Frames")
    st.pyplot(plot_crowding(crowding))

    fig = plot_crowding_timeline(crowding, date_filter="2022-01-09")
    if fig:
        st.subheader("Crowding Score over Time — Jan 9")
        st.pyplot(fig)

    st.subheader("Alert Frame Details")
    alert_rows = [
        {
            "Frame": c["frame_idx"],
            "Score": round(c["score"], 3),
            "Density": round(c["density"], 3),
            "Overlap Rate": round(c["overlap_rate"], 3),
            "Count": c["count"],
        }
        for c in crowding if c["alert"]
    ]
    if alert_rows:
        st.dataframe(alert_rows, use_container_width=True)
    else:
        st.success("No crowding alerts detected.")
