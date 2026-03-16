# YOLO Object Detection & Segmentation App

A sleek, fast, and modern web application built with Streamlit and YOLOv8 for detecting and segmenting objects in images.

## Features

- **Upload Images**: Supports `.jpg`, `.jpeg`, `.png`, and `.webp` formats.
- **YOLOv8 Segmentation**: Accurately outlines objects and provides precise visual masks.
- **Summary Metrics**: Instantly see a breakdown and count of all detected objects in an easy-to-read table.
- **Fast and Modern**: Backed by `uv` for lightning-fast Python dependency management.

## Project Structure

```text
yolo-object-detection-app/
├── app.py                 # Main Streamlit web frontend
├── src/
│   ├── __init__.py        
│   └── yolo_helper.py     # Backend YOLO inference and image processing logic
├── pyproject.toml         # uv project configuration and dependencies
└── README.md
```

## Prerequisites

- [Python](https://www.python.org/downloads/) 3.9 or higher.
- [uv](https://docs.astral.sh/uv/) for incredibly fast package management.

### Installing `uv` (if you don't have it)

If you haven't installed `uv` yet, you can do so easily:
```bash
pip install uv
```

## Getting Started

Follow these steps to set up the project locally:

1. **Clone or Download the Repository** and navigate into the project directory:
   ```bash
   cd yolo-object-detection-app
   ```

2. **Sync the project and install dependencies**:
   ```bash
   uv sync
   ```
   *(This will automatically ensure the virtual environment using `pyproject.toml` has required packages like `streamlit`, `ultralytics`, `opencv`, and `pillow`)*

3. **Run the Application**:
   ```bash
   uv run streamlit run app.py
   ```

4. **View in Browser**:
   Open the Local URL provided in your terminal (usually `http://localhost:8501`) to interact with your Object Detection app!

## Note on First Run
The first time you upload an image, the backend will automatically download the `yolov8n-seg.pt` model weights (approx. 6-7 MB) from Ultralytics to process the image. Subsequent runs will be much faster.