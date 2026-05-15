import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def load_frames(frames_dir: str) -> list[Path]:
    """Load image frames sorted by filename (timestamp order)."""
    frames_dir = Path(frames_dir)
    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in exts])
    return frames


def run_tracking(frames_dir: str, model_path: str, conf: float = 0.3) -> list[dict]:
    """
    Run YOLO tracking on a sequence of frames.
    Returns a list of per-frame results:
      [{"frame_idx": int, "frame_path": str, "tracks": [{"id": int, "box": [x1,y1,x2,y2], "conf": float}]}]
    """
    model = YOLO(model_path)
    frames = load_frames(frames_dir)

    all_results = []
    for idx, frame_path in enumerate(frames):
        result = model.track(
            source=str(frame_path),
            conf=conf,
            persist=True,   # keep track IDs consistent across frames
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        tracks = []
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for box, tid, c in zip(boxes, ids, confs):
                tracks.append({
                    "id": int(tid),
                    "box": box.tolist(),
                    "conf": float(c),
                })

        all_results.append({
            "frame_idx": idx,
            "frame_path": str(frame_path),
            "tracks": tracks,
        })

    return all_results


def draw_tracks(frame_result: dict) -> np.ndarray:
    """Draw bounding boxes and track IDs on a frame, return BGR image."""
    img = cv2.imread(frame_result["frame_path"])

    colors = {}
    for track in frame_result["tracks"]:
        tid = track["id"]
        if tid not in colors:
            np.random.seed(tid * 17)
            colors[tid] = tuple(int(x) for x in np.random.randint(50, 255, 3))

        x1, y1, x2, y2 = (int(v) for v in track["box"])
        color = colors[tid]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"Pig #{tid}  {track['conf']:.2f}"
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    count = len(frame_result["tracks"])
    cv2.putText(img, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return img


def save_tracked_video(all_results: list[dict], output_path: str, fps: float = 5.0):
    """Save all tracked frames as a video."""
    if not all_results:
        return

    sample = cv2.imread(all_results[0]["frame_path"])
    h, w = sample.shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for frame_result in all_results:
        img = draw_tracks(frame_result)
        out.write(img)

    out.release()
