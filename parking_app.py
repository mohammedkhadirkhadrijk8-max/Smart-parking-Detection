import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
import urllib.request

st.set_page_config(page_title="AVP Smart Parking Telemetry", layout="wide")

# ==============================
# 1. AUTO-DOWNLOAD REAL PARKING DATASET
# ==============================
DATASET_URLS = {
    "Aerial Parking Lot (Standard Benchmark)": "https://raw.githubusercontent.com/harshbafnaa/car-parking-detection/main/carPark.mp4",
    "Alternative Mirror (carPark.mp4)": "https://raw.githubusercontent.com/sayanbarik/parking-space-detection/main/carPark.mp4"
}

def download_sample_video(url, save_path="carPark.mp4"):
    try:
        urllib.request.urlretrieve(url, save_path)
        return os.path.exists(save_path) and os.path.getsize(save_path) > 10000
    except Exception:
        return False

# ==============================
# 2. LOAD AI MODEL
# ==============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ==============================
# 3. PAIRED BACK-TO-BACK PARKING BAY GENERATOR
# ==============================
def generate_paired_parking_bays(width, height, num_pairs, rows_per_col, margin_left, margin_right, margin_top, margin_bottom, aisle_gap, col_offsets):
    """
    Generates realistic back-to-back paired parking bays:
    [Col 1 | Col 2] <-- Aisle --> [Col 3 | Col 4] <-- Aisle --> [Col 5 | Col 6]
    Matches real-world parking lot architecture and carPark.mp4 geometry.
    """
    bays = []
    total_cols = num_pairs * 2
    usable_w = width * (1.0 - margin_left - margin_right)
    usable_h = height * (1.0 - margin_top - margin_bottom)

    num_aisles = max(num_pairs - 1, 0)
    bay_w = (usable_w - num_aisles * aisle_gap) / total_cols
    bay_h = usable_h / rows_per_col

    slot_num = 1
    curr_x = width * margin_left

    for pair_idx in range(num_pairs):
        for side in range(2):  # 0 = Left column of pair, 1 = Right column of pair
            col_idx = pair_idx * 2 + side
            y_shift = col_offsets[col_idx] if col_idx < len(col_offsets) else 0

            x1 = int(curr_x)
            x2 = int(curr_x + bay_w)

            for r in range(rows_per_col):
                y1 = int(height * margin_top + r * bay_h + y_shift + 2)
                y2 = int(height * margin_top + (r + 1) * bay_h + y_shift - 2)
                bay_id = f"P{slot_num:02d}"
                bays.append((bay_id, x1, y1, x2, y2))
                slot_num += 1

            curr_x += bay_w
        # Add driving aisle ONLY after every pair of back-to-back columns
        curr_x += aisle_gap

    return bays

def compute_iou_and_overlap(bay_box, car_box):
    bx1, by1, bx2, by2 = bay_box
    cx1, cy1, cx2, cy2 = car_box

    inter_x1 = max(bx1, cx1)
    inter_y1 = max(by1, cy1)
    inter_x2 = min(bx2, cx2)
    inter_y2 = min(by2, cy2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    bay_area = max((bx2 - bx1) * (by2 - by1), 1)
    return inter_area / bay_area

def estimate_asphalt_reference(frame, bays):
    """Samples median brightness/color of driving aisles to calibrate empty ground."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.median(gray))

def process_parking_telemetry(
    frame, results, bays, pixel_thresh, iou_thresh,
    show_valet_path, show_heatmap, bay_timers, fps, frame_idx
):
    height, width = frame.shape[:2]
    overlay = frame.copy()

    # 1. Multi-Cue Computer Vision Feature Maps (Catches white, silver, red, AND dark black cars!)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    # Cue A: Adaptive Edge & Texture Map
    thresh_map = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 14
    )
    median = cv2.medianBlur(thresh_map, 5)
    dilated_edges = cv2.dilate(median, np.ones((3, 3), np.uint8), iterations=1)

    # Cue B: Asphalt Intensity Deviation (Catches glossy black/dark cars with smooth roofs)
    asphalt_luma = estimate_asphalt_reference(frame, bays)

    # 2. Extract Top-Down YOLOv8 Vehicle Detections
    detected_vehicles = []
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Filter out giant false boxes covering whole lot
                if (x2 - x1) < width * 0.25 and (y2 - y1) < height * 0.25:
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    detected_vehicles.append((x1, y1, x2, y2, conf, track_id))

    free_slots = []
    occupied_count = 0

    # 3. Evaluate Every Parking Bay Using 3-Signal Fusion
    for bay_id, bx1, by1, bx2, by2 in bays:
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(width, bx2), min(height, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue

        # Inner core crop (ignores painted white border lines on edges of slot!)
        pad_x = max(int((bx2 - bx1) * 0.12), 2)
        pad_y = max(int((by2 - by1) * 0.14), 2)
        ix1, iy1 = min(bx1 + pad_x, bx2 - 1), min(by1 + pad_y, by2 - 1)
        ix2, iy2 = max(bx2 - pad_x, ix1 + 1), max(by2 - pad_y, iy1 + 1)

        # Signal 1: Inner-bay Edge Density
        crop_edges = dilated_edges[iy1:iy2, ix1:ix2]
        edge_pixels = cv2.countNonZero(crop_edges)
        inner_area = max((ix2 - ix1) * (iy2 - iy1), 1)
        edge_score = int((edge_pixels / inner_area) * 5000)

        # Signal 2: Dark/Bright Vehicle Contrast vs Empty Asphalt
        crop_gray = gray[iy1:iy2, ix1:ix2]
        mean_luma = float(np.mean(crop_gray))
        luma_diff = abs(mean_luma - asphalt_luma)
        is_dark_or_bright_car = (luma_diff > 34.0) and (edge_score > pixel_thresh * 0.45)

        # Signal 3: YOLOv8 Deep Learning Overlap
        max_overlap = 0.0
        for (cx1, cy1, cx2, cy2, conf, tid) in detected_vehicles:
            overlap = compute_iou_and_overlap((bx1, by1, bx2, by2), (cx1, cy1, cx2, cy2))
            if overlap > max_overlap:
                max_overlap = overlap

        is_occupied = (
            (max_overlap >= iou_thresh)
            or (edge_score >= pixel_thresh)
            or is_dark_or_bright_car
        )

        cx_bay = int((bx1 + bx2) / 2)
        cy_bay = int((by1 + by2) / 2)

        if is_occupied:
            occupied_count += 1
            if bay_id not in bay_timers:
                bay_timers[bay_id] = frame_idx
            dwell_sec = (frame_idx - bay_timers[bay_id]) / max(fps, 1.0)

            fill_color = (45, 45, 220)
            border_color = (75, 75, 255)
            status_text = f"{bay_id} | {dwell_sec:.1f}s"
        else:
            if bay_id in bay_timers:
                del bay_timers[bay_id]
            free_slots.append((bay_id, bx1, by1, bx2, by2, cx_bay, cy_bay, edge_score))

            fill_color = (75, 205, 75)
            border_color = (110, 245, 110)
            status_text = f"{bay_id} FREE"

        if show_heatmap:
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), fill_color, -1)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_color, 1, cv2.LINE_AA)
        tick = 4
        cv2.line(frame, (bx1, by1), (bx1 + tick, by1), border_color, 2, cv2.LINE_AA)
        cv2.line(frame, (bx1, by1), (bx1, by1 + tick), border_color, 2, cv2.LINE_AA)
        cv2.line(frame, (bx2, by2), (bx2 - tick, by2), border_color, 2, cv2.LINE_AA)
        cv2.line(frame, (bx2, by2), (bx2, by2 - tick), border_color, 2, cv2.LINE_AA)

        cv2.putText(
            frame, status_text, (bx1 + 3, by2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.31, (245, 245, 245), 1, cv2.LINE_AA
        )

    # 4. Draw Top-Down AI Vehicle Brackets
    for (cx1, cy1, cx2, cy2, conf, tid) in detected_vehicles:
        v_cx, v_cy = int((cx1 + cx2) / 2), int((cy1 + cy2) / 2)
        cv2.circle(frame, (v_cx, v_cy), 2, (255, 225, 60), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 210, 50), 1, cv2.LINE_AA)

    # 5. Autonomous Valet Guidance Line (Routes through the Driving Aisle!)
    best_slot_str = "LOT FULL"
    if len(free_slots) > 0:
        # Entry point in the main driving aisle
        entry_pt = (int(width * 0.35), height - 12)
        free_slots.sort(key=lambda s: np.hypot(s[5] - entry_pt[0], s[6] - entry_pt[1]))
        best_slot = free_slots[0]
        best_id, s_x1, s_y1, s_x2, s_y2, target_x, target_y, _ = best_slot
        best_slot_str = f"SLOT {best_id}"

        cv2.rectangle(frame, (s_x1 - 2, s_y1 - 2), (s_x2 + 2, s_y2 + 2), (255, 255, 0), 2, cv2.LINE_AA)

        if show_valet_path:
            # Find nearest driving aisle X coordinate next to target slot
            aisle_x = s_x2 + 28 if target_x < width * 0.5 else s_x1 - 28
            wp1 = (aisle_x, height - 12)
            wp2 = (aisle_x, target_y)
            cv2.line(overlay, entry_pt, wp1, (255, 245, 0), 2, cv2.LINE_AA)
            cv2.line(overlay, wp1, wp2, (255, 245, 0), 2, cv2.LINE_AA)
            cv2.line(overlay, wp2, (target_x, target_y), (255, 245, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (target_x, target_y), 4, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(
                frame, f"AVP -> {best_id}", (s_x1, max(s_y1 - 5, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 110), 1, cv2.LINE_AA
            )

    frame = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)

    # 6. Top Industrial Telemetry HUD Banner
    total_bays = len(bays)
    free_count = len(free_slots)
    occupancy_rate = int((occupied_count / max(total_bays, 1)) * 100)

    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (10, 10), (width - 10, 48), (15, 20, 18), -1)
    frame = cv2.addWeighted(hud_bg, 0.78, frame, 0.22, 0)

    hud_text = (
        f"AVP TELEMETRY  |  FREE: {free_count}/{total_bays}  |  "
        f"OCCUPANCY: {occupancy_rate}%  |  AI VEHICLES: {len(detected_vehicles)}  |  "
        f"VALET GUIDE: {best_slot_str}"
    )
    cv2.putText(frame, hud_text, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 250, 180), 1, cv2.LINE_AA)

    return frame, free_count, occupied_count, occupancy_rate, best_slot_str

# ==============================
# 4. STREAMLIT UI & CONTROLS
# ==============================
st.sidebar.title("🅿️ AVP Smart Parking Controls")
st.sidebar.write("Calibrated **Paired-Bay Geometry + 3-Signal Fusion** Valet Engine.")

st.sidebar.subheader("1. Dataset Quick-Loader")
if st.sidebar.button("📥 Auto-Download Benchmark Dataset (carPark.mp4)"):
    with st.sidebar.status("Downloading aerial parking lot dataset..."):
        success = False
        for name, url in DATASET_URLS.items():
            if download_sample_video(url, "carPark.mp4"):
                success = True
                break
        if success:
            st.sidebar.success("✅ `carPark.mp4` downloaded!")
        else:
            st.sidebar.error("Download blocked by network. Please upload any MP4 manually.")

st.sidebar.subheader("2. Paired Parking Matrix Calibration")
num_pairs = st.sidebar.slider("Back-to-Back Column Pairs", 1, 5, 3)
rows_per_col = st.sidebar.slider("Slots per Column", 6, 15, 12)
margin_left = st.sidebar.slider("Left Lot Margin", 0.01, 0.20, 0.04, 0.005)
margin_right = st.sidebar.slider("Right Lot Margin", 0.01, 0.20, 0.08, 0.005)
margin_top = st.sidebar.slider("Top Lot Margin", 0.02, 0.20, 0.08, 0.005)
margin_bottom = st.sidebar.slider("Bottom Lot Margin", 0.02, 0.20, 0.08, 0.005)
aisle_gap = st.sidebar.slider("Driving Aisle Width (px)", 30, 160, 96, 2)

st.sidebar.subheader("3. AI & Dark-Car Sensitivity")
pixel_thresh = st.sidebar.slider("Inner-Slot Edge Threshold", 300, 1600, 720, 20)
iou_thresh = st.sidebar.slider("YOLOv8 Vehicle Overlap (IoU)", 0.10, 0.60, 0.18, 0.02)
show_valet_path = st.sidebar.checkbox("🧭 Show Autonomous Valet Guidance Path", value=True)
show_heatmap = st.sidebar.checkbox("🟩 Show Low-Contrast Slot Carpet", value=True)

st.title("🅿️ Automated Parking Slot Detection & Valet Guidance (AVP)")
st.write(
    "Calibrated **Back-to-Back Paired Parking Matrix** + **YOLOv8 Top-Down Enhancer** + "
    "**Dark-Car Asphalt Contrast Fusion** + **Aisle-Aware Valet Pathfinding**."
)

m1, m2, m3, m4 = st.columns(4)
metric_free = m1.empty()
metric_occ = m2.empty()
metric_rate = m3.empty()
metric_valet = m4.empty()

use_default = os.path.exists("carPark.mp4") and os.path.getsize("carPark.mp4") > 10000
video_source_mode = st.radio(
    "Select Video Source:",
    ["🎬 Use Downloaded Benchmark (carPark.mp4)", "📁 Upload Custom Parking Video"],
    index=0 if use_default else 1
)

input_video_path = None
if video_source_mode == "🎬 Use Downloaded Benchmark (carPark.mp4)":
    if os.path.exists("carPark.mp4"):
        input_video_path = "carPark.mp4"
        st.info("✅ Ready to run using `carPark.mp4` benchmark dataset!")
    else:
        st.warning("Click **'📥 Auto-Download Benchmark Dataset'** in the left sidebar first, or upload a video!")
else:
    uploaded_file = st.file_uploader("Upload an overhead or angled parking lot video (MP4/AVI)", type=["mp4", "avi"])
    if uploaded_file is not None:
        input_video_path = "uploaded_parking.mp4"
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success("Custom parking video uploaded!")

if input_video_path and st.button("🚀 Start Smart Parking Telemetry"):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Slight vertical stagger offsets tuned for carPark.mp4 paired rows
    col_offsets = [0, 0, -4, -4, -8, -8]
    bays = generate_paired_parking_bays(
        width, height, num_pairs, rows_per_col,
        margin_left, margin_right, margin_top, margin_bottom, aisle_gap, col_offsets
    )

    os.makedirs("output", exist_ok=True)
    output_path = "output/smart_parking_avp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    live_frame_box = st.empty()
    bay_timers = {}
    frame_idx = 0

    # CLAHE contrast booster for top-down vehicle detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    with st.spinner("Running Calibrated Paired-Bay AVP Telemetry..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Enhance top-down contrast slightly before feeding to YOLOv8
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            enhanced_for_yolo = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

            results = model.track(enhanced_for_yolo, conf=0.12, imgsz=960, persist=True, verbose=False)

            annotated, free_c, occ_c, rate_c, best_slot = process_parking_telemetry(
                frame, results, bays, pixel_thresh, iou_thresh,
                show_valet_path, show_heatmap, bay_timers, fps, frame_idx
            )

            out.write(annotated)

            if frame_idx % 5 == 0 or frame_idx == 1:
                metric_free.metric("🟢 Available Slots", f"{free_c} / {len(bays)}")
                metric_occ.metric("🔴 Occupied Slots", f"{occ_c}")
                metric_rate.metric("📊 Lot Utilization", f"{rate_c}%")
                metric_valet.metric("🧭 Valet Target", best_slot)

                rgb_preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                live_frame_box.image(
                    rgb_preview,
                    caption=f"Live AVP Telemetry Feed (Frame {frame_idx}/{total_frames})",
                    channels="RGB"
                )
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.progress(1.0)

    if os.path.exists(output_path):
        st.success("✅ Smart Parking Telemetry Video Processed Successfully!")
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Processed AVP Telemetry Video",
                data=f,
                file_name="Smart_Parking_AVP_Telemetry.mp4",
                mime="video/mp4"
            )
