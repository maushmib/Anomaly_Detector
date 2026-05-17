import os
import subprocess
import csv
import folium

from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# Preview Map using only GPS CSV
# ==========================================

@app.route("/preview_map", methods=["POST"])
def preview_map():

    csv_file = request.files["csv"]

    csv_path = os.path.join(
        UPLOAD_FOLDER,
        "preview_gps.csv"
    )

    map_path = os.path.join(
        UPLOAD_FOLDER,
        "preview_map.html"
    )

    csv_file.save(csv_path)

    gps_data = []

    with open(csv_path) as f:

        reader = csv.DictReader(f)

        for row in reader:

            gps_data.append({
                "lat": float(row["lat"]),
                "lon": float(row["lon"])
            })

    lats = [g["lat"] for g in gps_data]
    lons = [g["lon"] for g in gps_data]

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=19,
        tiles="OpenStreetMap"
    )

    path_points = []

    for g in gps_data:

        lat = g["lat"]
        lon = g["lon"]

        path_points.append([lat, lon])

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="blue",
            fill=True,
            fill_color="blue",
            popup=f"""
            Latitude: {lat}<br>
            Longitude: {lon}
            """
        ).add_to(m)

    folium.PolyLine(
        path_points,
        color="blue",
        weight=3
    ).add_to(m)

    m.save(map_path)

    return send_file(
        map_path,
        mimetype="text/html"
    )


# ==========================================
# Full Detection API
# ==========================================

@app.route("/process", methods=["POST"])
def process_files():

    video = request.files["video"]
    csv_file = request.files["csv"]

    video_path = os.path.join(
        UPLOAD_FOLDER,
        "video.mp4"
    )

    csv_path = os.path.join(
        UPLOAD_FOLDER,
        "gps.csv"
    )

    output_path = os.path.join(
        UPLOAD_FOLDER,
        "map.html"
    )

    video.save(video_path)

    csv_file.save(csv_path)

    result = subprocess.run(
    [
        "python",
        "anomaly.py",
        video_path,
        csv_path,
        output_path
    ],
    capture_output=True,
    text=True
    )

    print(result.stdout)
    print(result.stderr)

    return send_file(
        output_path,
        mimetype="text/html"
    )


if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )