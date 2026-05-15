# Pig Monitoring & Behavior Analysis

A computer vision pipeline for pig welfare monitoring — detects, tracks, and analyzes pig behavior from surveillance footage using YOLOv8 + ByteTrack, with activity heatmaps, crowding alerts, and time-series analysis.

**Live Demo:** [pig-moinitoring-evjwctafly2rtuuhufnnnj.streamlit.app](https://pig-moinitoring-evjwctafly2rtuuhufnnnj.streamlit.app/)

---

## Features

- **Multi-object Tracking** — YOLOv8 detection + ByteTrack assigns consistent IDs to each pig across frames
- **Pig Count Analysis** — per-frame count statistics with time-series visualization
- **Activity Heatmap** — Gaussian-weighted heatmap showing where pigs spend most time; supports per-date and morning vs. evening comparison
- **Crowding Detection** — combined density + overlap score with configurable alert threshold
- **Interactive Dashboard** — Streamlit web app with sample data or custom image upload

## Key Results

- Detection model: YOLOv8n fine-tuned on pig surveillance footage, **mAP50 = 99.2%**
- Behavioral insight: evening sessions show pigs concentrated near the feeding trough; morning sessions show broader activity across two regions

## Project Structure

```
pig-tracking/
├── app.py                  # Streamlit dashboard
├── pipeline.py             # Batch analysis script
├── modules/
│   ├── tracker.py          # YOLOv8 + ByteTrack tracking
│   ├── counter.py          # Count statistics and time-series plots
│   ├── heatmap.py          # Activity heatmap generation
│   ├── crowding.py         # Crowding score and alerts
│   └── utils.py            # Timestamp parsing and grouping
├── models/best.pt          # Fine-tuned YOLOv8n weights
├── data/samples/           # Sample pig images for demo
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the app, select **Use sample data**, and click **Run Analysis**.

## Tech Stack

Python · YOLOv8 · ByteTrack · OpenCV · Streamlit · Matplotlib

## Background

This project is part of a university–industry research collaboration on intelligent pig farming in China, focused on precision monitoring of livestock environments and real-time behavioral analysis to optimize production strategies.
