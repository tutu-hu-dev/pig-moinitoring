import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_heatmap(tracking_results: list[dict], ref_image_path: str) -> np.ndarray:
    """Accumulate bounding box centers across frames into a heatmap."""
    ref = cv2.imread(ref_image_path)
    h, w = ref.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for frame in tracking_results:
        for track in frame["tracks"]:
            x1, y1, x2, y2 = track["box"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            sigma_x = max(int(x2 - x1) // 4, 1)
            sigma_y = max(int(y2 - y1) // 4, 1)
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            blob = np.exp(-((xx - cx) ** 2 / (2 * sigma_x ** 2) + (yy - cy) ** 2 / (2 * sigma_y ** 2)))
            heatmap += blob

    return heatmap


def _overlay(heatmap: np.ndarray, ref_rgb: np.ndarray) -> np.ndarray:
    normalized = heatmap / heatmap.max()
    colored = (plt.cm.jet(normalized)[:, :, :3] * 255).astype(np.uint8)
    return cv2.addWeighted(ref_rgb, 0.5, colored, 0.5, 0)


def plot_heatmap(heatmap: np.ndarray, ref_image_path: str, save_path: str = None) -> plt.Figure:
    """Overlay heatmap on reference image."""
    ref_rgb = cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB)
    overlay = _overlay(heatmap, ref_rgb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(ref_rgb)
    axes[0].set_title("Reference Frame")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Activity Heatmap (all frames)")
    axes[1].axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, heatmap.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1], fraction=0.03, label="Activity Density")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_grouped_heatmaps(
    groups: dict[str, list[dict]],
    ref_image_path: str,
    title_prefix: str = "",
    save_path: str = None,
) -> plt.Figure:
    """Plot one heatmap per group side by side for comparison."""
    labels = list(groups.keys())
    n = len(labels)
    ref_rgb = cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB)
    h, w = ref_rgb.shape[:2]

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        frames = groups[label]
        if not frames:
            ax.set_title(f"{title_prefix}{label}\n(no data)")
            ax.axis("off")
            continue
        heatmap = build_heatmap(frames, ref_image_path)
        overlay = _overlay(heatmap, ref_rgb)
        ax.imshow(overlay)
        ax.set_title(f"{title_prefix}{label}\n({len(frames)} frames)")
        ax.axis("off")

    fig.suptitle("Heatmap Comparison", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
