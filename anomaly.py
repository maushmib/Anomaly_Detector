import os
import cv2
import numpy as np
import csv
import torch
import sys
import datetime
import folium

from math import ceil, tan, radians
from folium import Polygon
from pyproj import Transformer

from model import PatchCNN, PATCH_SIZE, STRIDE

# =========================
# INPUTS
# =========================

video_path = sys.argv[1]
gps_log_path = sys.argv[2]
output_map_path = sys.argv[3]

# =========================
# SETTINGS
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

target_fps = 0.3
cell_size_m = 1

HYBRID_RATIO_THRESHOLD = 0.03

# DRONE PARAMETERS (ASSUMED REAL-WORLD)
ALTITUDE_M = 3
FOV_H = 70
FOV_V = 50

# compute footprint size
footprint_w = 2 * ALTITUDE_M * tan(radians(FOV_H / 2))
footprint_h = 2 * ALTITUDE_M * tan(radians(FOV_V / 2))

# =========================
# GPS INTERPOLATION
# =========================

def interpolate_gps(gps_data, target_time):

    for i in range(len(gps_data) - 1):

        t1 = gps_data[i]["time"]
        t2 = gps_data[i + 1]["time"]

        if t1 <= target_time <= t2:

            ratio = (target_time - t1).total_seconds() / (t2 - t1).total_seconds()

            lat = gps_data[i]["lat"] + ratio * (gps_data[i + 1]["lat"] - gps_data[i]["lat"])
            lon = gps_data[i]["lon"] + ratio * (gps_data[i + 1]["lon"] - gps_data[i]["lon"])

            return lat, lon

    return gps_data[-1]["lat"], gps_data[-1]["lon"]

# =========================
# FRAME EXTRACTION
# =========================

cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, int(original_fps / target_fps))

frames = []
timestamps = []

count = 0
frame_no = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:

        frames.append(frame)
        timestamps.append(count / original_fps)

        frame_no += 1

    count += 1

cap.release()

print(f"Extracted {len(frames)} frames")

# =========================
# READ GPS
# =========================

gps_data = []

with open(gps_log_path) as f:

    reader = csv.DictReader(f)

    for row in reader:

        gps_data.append({
            "time": datetime.datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S.%f"),
            "lat": float(row["lat"]),
            "lon": float(row["lon"])
        })

# =========================
# MAP SETUP
# =========================

gps_points = [(g["lat"], g["lon"]) for g in gps_data]

center_lat = np.mean([p[0] for p in gps_points])
center_lon = np.mean([p[1] for p in gps_points])

m = folium.Map(location=[center_lat, center_lon], zoom_start=20)

folium.PolyLine(gps_points, color="blue").add_to(m)

# =========================
# TRANSFORMERS
# =========================

to_xy = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
to_latlon = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)

# =========================
# MODEL
# =========================

model = PatchCNN().to(DEVICE)
model.load_state_dict(torch.load("patch_cnn_model.pth", map_location=DEVICE))
model.eval()

# =========================
# GRID
# =========================

xy_points = [to_xy.transform(lon, lat) for lat, lon in gps_points]

min_x = min(p[0] for p in xy_points)
max_x = max(p[0] for p in xy_points)
min_y = min(p[1] for p in xy_points)
max_y = max(p[1] for p in xy_points)

grid_cols = max(1, ceil((max_x - min_x) / cell_size_m))
grid_rows = max(1, ceil((max_y - min_y) / cell_size_m))

grid = np.zeros((grid_rows, grid_cols))

anomaly_infos = []

# =========================
# PROCESS FRAMES
# =========================

for i, frame in enumerate(frames):

    frame_time = gps_data[0]["time"] + datetime.timedelta(seconds=timestamps[i])
    lat, lon = interpolate_gps(gps_data, frame_time)

    x, y = to_xy.transform(lon, lat)

    frame = cv2.resize(frame, (640, 360))

    h, w = frame.shape[:2]

    hybrid_count = 0
    total = 0

    # PATCH DETECTION
    for yy in range(0, h - PATCH_SIZE, STRIDE):
        for xx in range(0, w - PATCH_SIZE, STRIDE):

            patch = frame[yy:yy+PATCH_SIZE, xx:xx+PATCH_SIZE]
            patch = (patch / 255.0).astype(np.float32)
            patch = np.transpose(patch, (2, 0, 1))

            tensor = torch.tensor(patch).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = model(tensor).argmax(1).item()

            total += 1
            if pred == 1:
                hybrid_count += 1

    hybrid_ratio = hybrid_count / max(total, 1)

    # ANOMALY DETECTED
    if hybrid_ratio > HYBRID_RATIO_THRESHOLD:

        col = int((x - min_x) / cell_size_m)
        row = int((y - min_y) / cell_size_m)

        row = np.clip(row, 0, grid_rows - 1)
        col = np.clip(col, 0, grid_cols - 1)

        # FOOTPRINT SPREAD (REAL CAMERA AREA)
        fx = int(footprint_w / cell_size_m)
        fy = int(footprint_h / cell_size_m)

        for rr in range(row - fy//2, row + fy//2):
            for cc in range(col - fx//2, col + fx//2):

                if 0 <= rr < grid_rows and 0 <= cc < grid_cols:
                    grid[rr, cc] += hybrid_ratio

        dx = x - min_x
        dy = y - min_y

        info = {
            "row": row,
            "col": col,
            "lat": lat,
            "lon": lon,
            "forward_m": round(abs(dy), 2),
            "side_m": round(abs(dx), 2)
        }

        anomaly_infos.append(info)

        print("\nANOMALY DETECTED")
        print(f"Grid: ({row}, {col})")
        print(f"Move {info['forward_m']}m straight, {info['side_m']}m right")

# =========================
# DRAW GRID
# =========================

for r in range(grid_rows):
    for c in range(grid_cols):

        x1 = min_x + c * cell_size_m
        y1 = min_y + r * cell_size_m
        x2 = x1 + cell_size_m
        y2 = y1 + cell_size_m

        corners = [
            to_latlon.transform(x1, y1),
            to_latlon.transform(x2, y1),
            to_latlon.transform(x2, y2),
            to_latlon.transform(x1, y2)
        ]

        val = grid[r, c]

        if val > 0:
            color = "red"
            fill = True
        else:
            color = "white"
            fill = False

        Polygon(
            locations=[(lat, lon) for lon, lat in corners],
            color=color,
            fill=fill,
            fill_opacity=0.5
        ).add_to(m)

# =========================
# MARKERS
# =========================

for a in anomaly_infos:

    folium.Marker(
        location=[a["lat"], a["lon"]],
        popup=f"""
        ANOMALY<br>
        Grid ({a['row']},{a['col']})<br>
        Move {a['forward_m']}m straight<br>
        Move {a['side_m']}m right
        """,
        icon=folium.Icon(color="red")
    ).add_to(m)

# =========================
# SAVE MAP
# =========================

m.save(output_map_path)

print("Map saved successfully")