# VisionTrack 👁️📍
### Interactive YOLO-OBB Training, Inference & Visual Inspector for Pseudopupil Centroid Analytics

Welcome to **VisionTrack**, a full-stack, real-time interactive workbench designed to streamline the lifecycle of training and running **YOLO Oriented Bounding Box (OBB)** models on MATLAB (`.mat`) biological imaging data. It provides an elegant, real-time GUI wrapping underlying Python ML pipelines, facilitating dataset preparation, hyperparameter modeling, logging analysis, and high-fidelity interactive annotation or prediction validation.

This document serves as the primary technical hand-off and onboarding manual for developers and ML engineers taking over or continuing development of this workspace.

---

## 📂 Project Architecture Overview

VisionTrack functions as a highly modularized full-stack architecture:

```
├── server.ts                  # Express.js orchestrator (spawns tasks, parses files, processes REST APIs)
├── train_yolo_pseudopupil.py  # Python: Prep-pipeline, label formatting, and YOLO training script
├── infer_yolo_pseudopupil.py  # Python: MATLAB reading, localized image processing, and inference script
├── package.json               # Node.js dependencies, scripts, and dev server configurations
├── vite.config.ts             # Vite development configuration
├── src/                       # Frontend SPA (Single Page Application) React client
│   ├── App.tsx                # Client application hub & state engine
│   ├── components/            # Reusable core visual components
│   │   ├── VisionTrackInspectorPanel.tsx  # Dynamic interactive inspector and SVG manual refinement panel
│   │   ├── ModelPerformanceDashboard.tsx  # Interactive charts graphing epoch losses, accuracy, and metric gains
│   └── lib/                   # Utility classes & parses (e.g., CSV mapping, geometry calculations)
├── visible/                   # Base visible data directory
│   ├── mat_files/             # Workspace subfolders holding raw multi-channel MATLAB (.mat) files
│   ├── images/                # Dynamic PNG renders of MATLAB channels for UI rendering
│   └── labels/                # Standardized YOLO formatted target outputs
└── csv_outputs/               # Storage directory for compiled inference result CSV logs
```

---

## 🐍 Python ML Environment Setup

The core YOLO processing and dataset orchestration rely on Python libraries. Follow these steps to configure a stable local Python Conda environment for executing training and inference tasks.

### 1. Create a Dedicated Conda Environment
Using Anaconda or Miniconda, create a fresh sandbox environment with Python **3.10**:
```bash
conda create --name visiontrack python=3.10 -y
conda activate visiontrack
```

### 2. Install PyTorch with Hardware Acceleration
Configure PyTorch depending on your host machine's hardware capabilities:

* **For CUDA-enabled GPUs (NVIDIA Recommended):**
  ```bash
  # Install PyTorch with appropriate CUDA version (e.g., CUDA 11.8 or 12.1)
  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
  ```

* **For macOS (Apple Silicon MPS / CPU):**
  ```bash
  conda install pytorch torchvision -c pytorch -y
  ```

* **For CPU-only machines:**
  ```bash
  conda install pytorch torchvision cpuonly -c pytorch -y
  ```

### 3. Install Target Dependencies and Ultralytics
Install the necessary imaging, manipulation, and deep learning libraries via `pip`:
```bash
pip install ultralytics scikit-image opencv-python pandas numpy scipy pyyaml
```

*Installed Modules Description:*
* `ultralytics`: Official YOLO package, used for model loading, OBB target generation, val steps, and export metrics.
* `scikit-image` (`skimage`): Executes OTSU thresholding, local contrast enhancements, flood-fill algorithm mappings, and input normalization layers.
* `opencv-python` (`cv2`): Rapid high-speed channel extraction, format conversions, and saving image previews on disk.
* `pandas` & `numpy`: Multi-dimensional array structures and fast CSV/tabular telemetry outputs.
* `scipy`: Reads MATLAB `.mat` files and executes quick uniform filters.
* `pyyaml`: Dynamic generation and parsing of dataset `.yaml` files required by YOLO training.

---

## ⚡ Express Server & UI Workbench Setup

The UI interacts directly with the back-end using active port proxies to monitor logs, run tasks, and render high-resolution assets.

### 1. Install Node.js Dependencies
Ensure you have **Node.js (v18.0.0 or higher)** installed, then run:
```bash
npm install
```

### 2. Set Up Environment Variables
Create a `.env` file in the root directory (using `.env.example` as a template structure) if any external APIs or environments are configured.
```bash
cp .env.example .env
```

### 3. Start Development Mode
The Node development server spawns an interactive Express app integrated with the Vite middleware:
```bash
npm run dev
```
Once started, open **`http://localhost:3000`** in your browser.

### 4. Build & Run in Production
To bundle assets and compile TS files into static production bundles:
```bash
# Build Vite client assets and bundle server TS scripts
npm run build

# Boot CJS production build
npm run start
```

---

## ⚙️ Direct ML Execution & Parameter Details

For testing pipelines outside the UI framework, you can invoke core scripts directly from your activated terminal environment.

### 1. YOLO-OBB Model Training Pipeline (`train_yolo_pseudopupil.py`)
This script crawls the targeted directory, converts biological multi-channel MATLAB `.mat` workspace files into standardized annotated datasets, and triggers YOLO OBB modeling.

```bash
python train_yolo_pseudopupil.py \
  --epochs 100 \
  --batch 8 \
  --device cuda \
  --conda_env "visiontrack" \
  --selectedDir "workspace_v1"
```

* **`--epochs`**: Train epoch bounds (Default: `100`).
* **`--batch`**: Mini-batch size (Default: `8`).
* **`--device`**: Targeted hardware e.g. `cuda`, `cpu`, or `mps`.
* **`--conda_env`**: Selected host Conda shell context name to run child steps.
* **`--selectedDir`**: Targeted folder inside `visible/mat_files/` containing data to train on.

### 2. Automated Segmentation & Metric Inference (`infer_yolo_pseudopupil.py`)
Executes predictive evaluation on unannotated MATLAB trials, registering measurements, centroid mappings, and generating CSV reports.

```bash
python infer_yolo_pseudopupil.py \
  --version 8 \
  --size m \
  --device cpu \
  --data_dir "./visible/mat_files/workspace_v1" \
  --output "./csv_outputs/workspace_v1/results-yolob-obb.csv"
```

* **`--version`**: Target version of YOLO weights (e.g. `8` or `11`).
* **`--size`**: Model scale variant (`n`, `s`, `m`, `l`, `x`).
* **`--device`**: CPU or GPU engine target.
* **`--data_dir`**: Workspace directory with `.mat` targets to run inference on.
* **`--output`**: Target CSV file output location.

---

## 📝 Functional Feature Implementation Notes

### 📁 Dynamic Folder Segregation & Path Hierarchy
To avert confusion or dataset overlapping:
* **Directory Routing**: Dynamic selector states are persisted across both training, inference, and inspection workflows.
* **Inference Outputs**: Run results are isolated inside matching subdirectories (`csv_outputs/<selectedDir>/results-yolo-obb-...csv`).
* **Interactive CSV Scope Filters**:
  * Selecting **Default Workspace** limits results and telemetry visualizations strictly to root data files.
  * Adjusting the workspace dropdown automatically isolates files, thumbnails, and loaded inference logs to the target workspace folder.
