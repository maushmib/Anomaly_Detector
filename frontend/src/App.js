import { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {

  const [video, setVideo] = useState(null);
  const [csv, setCsv] = useState(null);
  const [result, setResult] = useState(null);

  const [loadingPreview, setLoadingPreview] = useState(false);
  const [loadingProcess, setLoadingProcess] = useState(false);

  const [error, setError] = useState("");

  // =========================
  // Validate File
  // =========================

  const validateFile = (file, type) => {

    if (!file) return false;

    if (type === "video") {
      return file.type.startsWith("video/");
    }

    if (type === "csv") {
      return file.name.endsWith(".csv");
    }

    return false;
  };

  // =========================
  // CSV Preview Map
  // =========================

  const handleCSV = async (file) => {

    setError("");

    if (!validateFile(file, "csv")) {
      setError("Please upload a valid CSV file");
      return;
    }

    setCsv(file);
    setLoadingPreview(true);

    const formData = new FormData();
    formData.append("csv", file);

    try {

      const res = await axios.post(
        "http://localhost:5000/preview_map",
        formData,
        { responseType: "text" }
      );

      const blob = new Blob([res.data], { type: "text/html" });
      const url = URL.createObjectURL(blob);

      setResult(url);

    } catch (err) {
      setError("Map preview failed. Check backend.");
    }

    setLoadingPreview(false);
  };

  // =========================
  // Full Processing
  // =========================

  const handleUpload = async () => {

    setError("");

    if (!video || !csv) {
      setError("Please select both video and CSV file");
      return;
    }

    if (!validateFile(video, "video")) {
      setError("Invalid video file");
      return;
    }

    setLoadingProcess(true);

    const formData = new FormData();
    formData.append("video", video);
    formData.append("csv", csv);

    try {

      const res = await axios.post(
        "http://localhost:5000/process",
        formData,
        { responseType: "text" }
      );

      const blob = new Blob([res.data], { type: "text/html" });
      const url = URL.createObjectURL(blob);

      setResult(url);

    } catch (err) {
      setError("Processing failed. Try again.");
    }

    setLoadingProcess(false);
  };

  // =========================
  // Reset System
  // =========================

  const handleReset = () => {
    setVideo(null);
    setCsv(null);
    setResult(null);
    setError("");
  };

  return (

    <div className="app">

      <header className="header">
        🌾 Paddy Field Anomaly Detection System
      </header>

      <div className="container">

        {/* LEFT PANEL */}
        <div className="card input-card">

          <h2>Upload Data</h2>

          {error && (
            <div className="error-box">
              {error}
            </div>
          )}

          <label className="input-label">
            Drone Video
            <input
              type="file"
              accept="video/*"
              onChange={(e) =>
                setVideo(e.target.files[0])
              }
            />
          </label>

          <label className="input-label">
            GPS CSV Log
            <input
              type="file"
              accept=".csv"
              onChange={(e) =>
                handleCSV(e.target.files[0])
              }
            />
          </label>

          <button
            className="btn"
            onClick={handleUpload}
            disabled={loadingProcess || loadingPreview}
          >
            {loadingProcess
              ? "Processing..."
              : "Run Detection"}
          </button>

          <button
            className="btn reset"
            onClick={handleReset}
          >
            Reset
          </button>

        </div>

        {/* RIGHT PANEL */}
        <div className="card output-card">

          <h2>Field Anomaly Map</h2>

          {loadingPreview && (
            <div className="placeholder">
              Loading preview map...
            </div>
          )}

          {!result && !loadingPreview && (
            <div className="placeholder">
              Upload data to view map
            </div>
          )}

          {result && (
            <iframe
              src={result}
              title="map"
              width="100%"
              height="650px"
              style={{
                border: "none",
                borderRadius: "12px"
              }}
            />
          )}

        </div>

      </div>

    </div>
  );
}