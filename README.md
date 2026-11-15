# Smart Parking Detection & Valet Guidance

An interactive Streamlit application that analyzes parking-lot video, estimates occupied and available spaces, and highlights a suggested valet target. It combines configurable parking-bay geometry with YOLOv8 vehicle tracking and image-based cues designed to help detect cars across a range of colors and contrast levels.

## Demo

[▶ Watch the processed parking-lot demo](output/smart_parking_avp.mp4)

The demo is also available as [`output/smart_parking_avp.mp4`](output/smart_parking_avp.mp4) in this repository.

## Features

- Configurable back-to-back parking rows, bay margins, and driving aisles
- YOLOv8 vehicle tracking combined with edge and asphalt-contrast cues
- Live available/occupied counts, lot utilization, and suggested valet target
- Optional valet guidance path and low-contrast slot overlay
- Processed MP4 output for download
- Benchmark-video download or upload of a custom MP4/AVI clip

## Run locally

Requires Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install streamlit ultralytics opencv-python numpy
streamlit run parking_app.py
```

On first run, Ultralytics may download the `yolov8n.pt` model weights if they are not already available.

## Use the app

1. In the sidebar, download the `carPark.mp4` benchmark or choose **Upload Custom Parking Video**.
2. Adjust the parking matrix, aisle, and detection settings to suit the footage.
3. Choose whether to show the valet guidance path and slot overlay.
4. Click **Start Smart Parking Telemetry**. The app displays live metrics and offers the processed video for download when complete.

Processed videos are written to `output/smart_parking_avp.mp4`; running the app again may replace that generated file.
