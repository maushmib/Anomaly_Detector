# =========================
# anomaly_realworld.py
# REAL-WORLD FIELD ANOMALY MAPPING
# =========================

import os
import cv2
import numpy as np
import csv
import torch
import sys
import datetime
import folium

from math import ceil, atan2, cos, sin
from folium import Polygon
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d

from model import PatchCNN, PATCH_SIZE, STRIDE

# =========================
# INPUTS
# =========================

video_path = sys.argv[1]
gps_log_path = sys.argv[2]
output_map_path = sys.argv[3]

# =========================
# OUTPUT FOLDERS
# =========================

output_frames_dir = "dataset/tests"
output_overlay_dir = "dataset/output"

os.makedirs(output_frames_dir, exist_ok=True)
os.makedirs(output_overlay_dir, exist_ok=True)

# =========================
# SETTINGS
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HIGHER FPS FOR REALISTIC MOVEMENT
target_fps = 2

# SMALLER GRID FOR BETTER FIELD DETAIL
cell_size_m = 0.5

HYBRID_RATIO_THRESHOLD = 0.03

# ESTIMATED GROUND WIDTH VISIBLE BY CAMERA
# adjust based on drone height/camera
visible_ground_width = 6

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

        frame_time = count / original_fps

        timestamps.append(frame_time)

        cv2.imwrite(
            os.path.join(
                output_frames_dir,
                f"frame{frame_no}.jpg"
            ),
            frame
        )

        frame_no += 1

    count += 1

cap.release()

print(f"\nExtracted {len(frames)} frames")

# =========================
# READ GPS DATA
# =========================

gps_data = []

with open(gps_log_path) as f:

    reader = csv.DictReader(f)

    for row in reader:

        t = datetime.datetime.strptime(
            row["time"],
            "%Y-%m-%d %H:%M:%S.%f"
        )

        gps_data.append({
            "time": t,
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "alt": float(row["alt"]),
            "speed": float(row["speed"])
        })

# =========================
# GPS POINTS
# =========================

gps_points = [
    (g["lat"], g["lon"])
    for g in gps_data
]

center_lat = np.mean([p[0] for p in gps_points])
center_lon = np.mean([p[1] for p in gps_points])

# =========================
# CREATE MAP
# =========================

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=22
)

# =========================
# SATELLITE VIEW
# =========================

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
    overlay=False,
    control=True
).add_to(m)

# =========================
# DRAW GPS PATH
# =========================

folium.PolyLine(
    gps_points,
    color="blue",
    weight=4,
    opacity=1
).add_to(m)

# =========================
# TRANSFORMERS
# =========================

transformer_to_xy = Transformer.from_crs(
    "epsg:4326",
    "epsg:3857",
    always_xy=True
)

transformer_to_latlon = Transformer.from_crs(
    "epsg:3857",
    "epsg:4326",
    always_xy=True
)

xy_points = []

for lat, lon in gps_points:

    x, y = transformer_to_xy.transform(
        lon,
        lat
    )

    xy_points.append((x, y))

# =========================
# GPS ARRAYS
# =========================

gps_x = np.array([p[0] for p in xy_points])
gps_y = np.array([p[1] for p in xy_points])

# =========================
# GPS SMOOTHING
# =========================

gps_x = gaussian_filter1d(gps_x, sigma=1)
gps_y = gaussian_filter1d(gps_y, sigma=1)

# =========================
# FIELD ROTATION
# =========================

start_x, start_y = gps_x[0], gps_y[0]
end_x, end_y = gps_x[-1], gps_y[-1]

dx = end_x - start_x
dy = end_y - start_y

theta = atan2(dy, dx)

print("\nField rotation angle:", np.degrees(theta))

# =========================
# ROTATE POINTS
# =========================

rotated_points = []

for x, y in zip(gps_x, gps_y):

    xr = (
        (x - start_x) * cos(-theta)
        -
        (y - start_y) * sin(-theta)
    )

    yr = (
        (x - start_x) * sin(-theta)
        +
        (y - start_y) * cos(-theta)
    )

    rotated_points.append((xr, yr))

# =========================
# GRID BOUNDS
# =========================

all_xr = [p[0] for p in rotated_points]
all_yr = [p[1] for p in rotated_points]

min_xr = min(all_xr)
max_xr = max(all_xr)

min_yr = min(all_yr)
max_yr = max(all_yr)

width = max_xr - min_xr
height = max_yr - min_yr

grid_cols = ceil(width / cell_size_m)
grid_rows = ceil(height / cell_size_m)

grid_cols = max(grid_cols, 1)
grid_rows = max(grid_rows, 1)

grid = np.zeros((grid_rows, grid_cols))

print(f"\nGrid Size: {grid_rows} x {grid_cols}")

# =========================
# LOAD MODEL
# =========================

model = PatchCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        "patch_cnn_model.pth",
        map_location=DEVICE
    )
)

model.eval()

# =========================
# GPS TIME ARRAYS
# =========================

gps_times = np.array([
    (
        g["time"] - gps_data[0]["time"]
    ).total_seconds()
    for g in gps_data
])

gps_alt = np.array([
    g["alt"]
    for g in gps_data
])

# =========================
# ANOMALY DETECTION
# =========================

path_points = []

for i, frame in enumerate(frames):

    print(f"\nProcessing frame {i+1}/{len(frames)}")

    frame = cv2.resize(frame, (640, 360))

    h, w = frame.shape[:2]

    pred_mask = np.zeros((h, w), dtype=np.uint8)

    hybrid_patch_count = 0

    frame_time = timestamps[i]

    # =========================
    # GPS INTERPOLATION
    # =========================

    interp_x = np.interp(
        frame_time,
        gps_times,
        gps_x
    )

    interp_y = np.interp(
        frame_time,
        gps_times,
        gps_y
    )

    interp_alt = np.interp(
        frame_time,
        gps_times,
        gps_alt
    )

    lon, lat = transformer_to_latlon.transform(
        interp_x,
        interp_y
    )

    path_points.append((lat, lon))

    # =========================
    # PATCH SCANNING
    # =========================

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):

        for x in range(0, w - PATCH_SIZE + 1, STRIDE):

            patch = frame[
                y:y+PATCH_SIZE,
                x:x+PATCH_SIZE
            ]

            patch = (patch / 255.0).astype(np.float32)

            patch = np.transpose(
                patch,
                (2, 0, 1)
            )

            tensor = torch.tensor(
                patch
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                pred = model(tensor).argmax(1).item()

            if pred == 1:

                hybrid_patch_count += 1

                cv2.circle(
                    pred_mask,
                    (
                        x + PATCH_SIZE//2,
                        y + PATCH_SIZE//2
                    ),
                    3,
                    1,
                    -1
                )

                # =========================
                # GROUND OFFSET ESTIMATION
                # =========================

                patch_center_x = x + PATCH_SIZE // 2

                offset_ratio = (
                    (patch_center_x - w/2)
                    / (w/2)
                )

                ground_offset = (
                    offset_ratio
                    * visible_ground_width
                    / 2
                )

                # =========================
                # ROTATED FIELD POSITION
                # =========================

                xr = (
                    (interp_x - start_x) * cos(-theta)
                    -
                    (interp_y - start_y) * sin(-theta)
                )

                yr = (
                    (interp_x - start_x) * sin(-theta)
                    +
                    (interp_y - start_y) * cos(-theta)
                )

                # apply offset
                xr += ground_offset

                col = int((xr - min_xr) / cell_size_m)
                row = int((yr - min_yr) / cell_size_m)

                row = np.clip(row, 0, grid_rows - 1)
                col = np.clip(col, 0, grid_cols - 1)

                grid[row, col] += 1

                # =========================
                # MOVEMENT INSTRUCTIONS
                # =========================

                forward_m = round(abs(yr), 2)
                side_m = round(abs(xr), 2)

                print("\nANOMALY FOUND")
                print("Lat:", lat)
                print("Lon:", lon)
                print("Grid:", row, col)

                print("\nMOVE INSTRUCTIONS")
                print(f"Move {forward_m}m straight")
                print(f"Move {side_m}m right")

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.9,
                    popup=f"""
                    <b>ANOMALY FOUND</b><br>
                    Frame: {i}<br>
                    Grid: ({row},{col})<br>
                    Altitude: {round(interp_alt,2)}m<br>
                    Move {forward_m}m straight<br>
                    Move {side_m}m right
                    """
                ).add_to(m)

    total_patches = (
        ((h - PATCH_SIZE)//STRIDE + 1)
        *
        ((w - PATCH_SIZE)//STRIDE + 1)
    )

    hybrid_ratio = hybrid_patch_count / total_patches

    print("Hybrid ratio:", hybrid_ratio)

    # =========================
    # SAVE OVERLAY
    # =========================

    overlay = frame.copy()

    overlay[pred_mask == 1] = [0, 0, 255]

    result = cv2.addWeighted(
        frame,
        0.7,
        overlay,
        0.5,
        0
    )

    cv2.imwrite(
        os.path.join(
            output_overlay_dir,
            f"frame{i}.png"
        ),
        result
    )

# =========================
# DRONE PATH
# =========================

folium.PolyLine(
    path_points,
    color="yellow",
    weight=5,
    opacity=1
).add_to(m)

# =========================
# DRAW FIELD GRID
# =========================

max_value = np.max(grid)

for row in range(grid_rows):

    for col in range(grid_cols):

        xr1 = min_xr + col * cell_size_m
        yr1 = min_yr + row * cell_size_m

        xr2 = xr1 + cell_size_m
        yr2 = yr1 + cell_size_m

        rotated_corners = [
            (xr1, yr1),
            (xr2, yr1),
            (xr2, yr2),
            (xr1, yr2)
        ]

        original_corners = []

        for xr, yr in rotated_corners:

            x = (
                xr * cos(theta)
                -
                yr * sin(theta)
            ) + start_x

            y = (
                xr * sin(theta)
                +
                yr * cos(theta)
            ) + start_y

            lon, lat = transformer_to_latlon.transform(
                x,
                y
            )

            original_corners.append((lat, lon))

        value = grid[row, col]

        # =========================
        # HEATMAP STYLE
        # =========================

        if value > 0:

            intensity = min(
                value / max_value,
                1
            )

            opacity = 0.2 + 0.6 * intensity

            color = "red"

        else:

            opacity = 0.08
            color = "green"

        Polygon(
            locations=original_corners,
            color="black",
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            popup=f"""
            Grid ({row},{col})
            <br>
            Count: {int(value)}
            """
        ).add_to(m)

# =========================
# LAYER CONTROL
# =========================

folium.LayerControl().add_to(m)

# =========================
# SAVE MAP
# =========================

m.save(output_map_path)

print("\nMap saved successfully!")

print("\nOriginal frames saved in:")
print(output_frames_dir)

print("\nOverlay frames saved in:")
print(output_overlay_dir)