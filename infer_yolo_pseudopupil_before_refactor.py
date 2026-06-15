# Pseudopupil Detection Pipeline — YOLO26m Bbox + Classical Refinement
# Runs inference asynchronously with live file progress reports to the Express backend.

import os
import sys
import glob
import time
import shutil
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import flood
import re
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SCIENTIFIC_NAME = "Aeschna_isoceles"
# By default, files are in `./visible/`
MAT_DIR         = "./visible"
PNG_DIR         = "./visible_png"

# Argument Parsing
parser = argparse.ArgumentParser(description="Run inference using YOLO + Classical refinement.")
parser.add_argument("--version", type=str, default="8", help="YOLO version (8, 11, 26 only)")
parser.add_argument("--size", type=str, default="m", help="YOLO size (n, s, m, l only)")
parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda)")
parser.add_argument("--output", type=str, default="", help="Custom output CSV file path")
args = parser.parse_args()

YOLO_VERSION = args.version
YOLO_SIZE    = args.size
YOLO_DEVICE  = args.device

# Check for model weights
PRIMARY_MODEL_PATH = f"./runs/pseudopupil_yolo{YOLO_VERSION}{YOLO_SIZE}_obb/weights/best.pt"
SECONDARY_MODEL_PATH = f"./runs/obb/runs/pseudopupil_yolo{YOLO_VERSION}{YOLO_SIZE}_obb/weights/best.pt"

# Resolve model path
if os.path.exists(PRIMARY_MODEL_PATH):
    MODEL_PATH = PRIMARY_MODEL_PATH
elif os.path.exists(SECONDARY_MODEL_PATH):
    MODEL_PATH = SECONDARY_MODEL_PATH
else:
    # Fallback to check if there is ANY best.pt inside directories
    candidate = None
    for root, dirs, files in os.walk("./runs"):
        if "best.pt" in files and f"yolo{YOLO_VERSION}{YOLO_SIZE}" in root:
            candidate = os.path.join(root, "best.pt")
            break
    MODEL_PATH = candidate if candidate else PRIMARY_MODEL_PATH

# Output CSV
if args.output:
    OUTPUT_CSV = args.output
else:
    OUTPUT_CSV = f"./results-yolo{YOLO_VERSION}{YOLO_SIZE}-obb-{SCIENTIFIC_NAME}.csv"

DEBUG           = True
DEBUG_DIR       = f"./debug_yolo{YOLO_VERSION}{YOLO_SIZE}_obb"

CONVERT_MAT_TO_PNG = True
KEEP_PNGS          = False

USE_BBOX_FOR_REFINE      = True
REFINE_BBOX_EXPAND_RATIO = 1.1   # expand bbox by this factor for the crop window
REFINE_MAX_SHIFT_RATIO   = 0.8   # max allowed refinement shift = bbox_diagonal × this

ANGLES_PER_FILE = 5   # imagesS0 through imagesS4

# YOLO inference settings
YOLO_CONF_THRESH = 0.25
YOLO_IOU_THRESH  = 0.45

# Stage 2 refinement defaults
REFINE_WINDOW_PX       = 80     # fallback fixed window radius
REFINE_MIN_BLOB_PX     = 3
REFINE_MAX_SHIFT_PX    = 40     # fallback fixed max shift
REFINE_FLOOD_TOLERANCE = 0.10
REFINE_BLUR_SIGMA      = 2

PNG_FLAT_LAYOUT = True
FLIP_HORIZONTAL = False

# ─────────────────────────────────────────────
# STAGE 1 — YOLO DETECTION (bbox only)
# ─────────────────────────────────────────────

def load_yolo_model(model_path: str) -> YOLO:
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights file not found at {model_path}", flush=True)
        print("💡 Please train the model configuration first.", flush=True)
        sys.exit(1)
    model = YOLO(model_path)
    print(f"Loaded YOLO{YOLO_VERSION}{YOLO_SIZE}-OBB model: {model_path}", flush=True)
    return model


def yolo_detect_best_box(model: YOLO, png_path: str):
    if not os.path.exists(png_path):
        return None, None, 0.0, None

    results = model.predict(
        source  = png_path,
        conf    = YOLO_CONF_THRESH,
        iou     = YOLO_IOU_THRESH,
        device  = YOLO_DEVICE,
        verbose = False,
    )

    if not results or results[0].obb is None or len(results[0].obb) == 0:
        return None, None, 0.0, None

    # OBB output: xywhr = [cx, cy, w, h, angle_rad] in pixel space
    xywhr = results[0].obb.xywhr.cpu().numpy()   # [N, 5]
    confs = results[0].obb.conf.cpu().numpy()     # [N]

    best_idx = int(np.argmax(confs))
    cx       = float(xywhr[best_idx, 0])
    cy       = float(xywhr[best_idx, 1])
    w_box    = float(xywhr[best_idx, 2])
    h_box    = float(xywhr[best_idx, 3])
    conf     = float(confs[best_idx])

    # Convert OBB centre+size to axis-aligned bbox for Stage 2 crop window
    bbox = (cx - w_box / 2.0, cy - h_box / 2.0,
            cx + w_box / 2.0, cy + h_box / 2.0)

    if FLIP_HORIZONTAL:
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            w_img = img.shape[1]
            cx    = w_img - cx
            bbox  = (w_img - bbox[2], bbox[1], w_img - bbox[0], bbox[3])

    return cx, cy, conf, bbox


def get_coarse_centre_from_yolo(model, png_path_i1, png_path_i2):
    cx_i1, cy_i1, conf_i1, bbox_i1 = yolo_detect_best_box(model, png_path_i1)
    cx_i2, cy_i2, conf_i2, bbox_i2 = yolo_detect_best_box(model, png_path_i2)

    if cx_i1 is None and cx_i2 is None:
        return None, None, 0.0, None, None
    if cx_i1 is None:
        return cx_i2, cy_i2, conf_i2, "i2", bbox_i2
    if cx_i2 is None:
        return cx_i1, cy_i1, conf_i1, "i1", bbox_i1
    if conf_i1 >= conf_i2:
        return cx_i1, cy_i1, conf_i1, "i1", bbox_i1
    return cx_i2, cy_i2, conf_i2, "i2", bbox_i2


# ─────────────────────────────────────────────
# .MAT FILE I/O
# ─────────────────────────────────────────────

def load_raw_I1(mat_path: str, angle_idx: int) -> np.ndarray:
    mat       = sio.loadmat(mat_path)
    imdat     = mat["imdat"]
    angle_key = f"imagesS{angle_idx}"
    I1_raw    = imdat[angle_key][0][0]["presetcapture"][0][0]["image"][0][0]
    return rescale_intensity(
        img_as_float(np.squeeze(I1_raw)), out_range=(0.0, 1.0)
    ).astype(np.float32)


def load_raw_I2(mat_path: str, angle_idx: int) -> np.ndarray:
    mat       = sio.loadmat(mat_path)
    imdat     = mat["imdat"]
    angle_key = f"imagesS{angle_idx}"
    I2_raw    = imdat[angle_key][0][0]["presetcapture"][0][0]["image"][0][1]
    return rescale_intensity(
        img_as_float(np.squeeze(I2_raw)), out_range=(0.0, 1.0)
    ).astype(np.float32)


def convert_mat_to_pngs(mat_path: str, angle_idx: int, out_dir: str) -> tuple:
    import cv2 as _cv2
    os.makedirs(out_dir, exist_ok=True)
    mat_stem = Path(mat_path).stem

    I1 = load_raw_I1(mat_path, angle_idx)
    I2 = load_raw_I2(mat_path, angle_idx)

    if FLIP_HORIZONTAL:
        I1 = I1[:, ::-1]
        I2 = I2[:, ::-1]

    path_i1 = os.path.join(out_dir, f"{mat_stem}_imagesS{angle_idx}_i1.png")
    path_i2 = os.path.join(out_dir, f"{mat_stem}_imagesS{angle_idx}_i2.png")
    _cv2.imwrite(path_i1, (I1 * 255.0).astype(np.uint8))
    _cv2.imwrite(path_i2, (I2 * 255.0).astype(np.uint8))
    return path_i1, path_i2


# ─────────────────────────────────────────────
# STAGE 2 — CLASSICAL BRIGHTNESS REFINEMENT
# ─────────────────────────────────────────────

def refine_with_brightness_centroid(I1_raw: np.ndarray,
                                     cx_coarse: float,
                                     cy_coarse: float,
                                     bbox: tuple = None) -> tuple:
    h, w = I1_raw.shape[:2]

    # Crop window
    if USE_BBOX_FOR_REFINE and bbox is not None:
        bx0, by0, bx1, by1 = bbox
        bbox_w   = bx1 - bx0
        bbox_h   = by1 - by0
        bbox_cx  = (bx0 + bx1) / 2.0
        bbox_cy  = (by0 + by1) / 2.0
        half_w   = (bbox_w * REFINE_BBOX_EXPAND_RATIO) / 2.0
        half_h   = (bbox_h * REFINE_BBOX_EXPAND_RATIO) / 2.0
        x0 = int(np.clip(bbox_cx - half_w, 0, w - 1))
        y0 = int(np.clip(bbox_cy - half_h, 0, h - 1))
        x1 = int(np.clip(bbox_cx + half_w, 0, w - 1))
        y1 = int(np.clip(bbox_cy + half_h, 0, h - 1))
        bbox_diag    = np.sqrt(bbox_w**2 + bbox_h**2)
        max_shift_px = bbox_diag * REFINE_MAX_SHIFT_RATIO
    else:
        x0 = int(np.clip(cx_coarse - REFINE_WINDOW_PX, 0, w - 1))
        y0 = int(np.clip(cy_coarse - REFINE_WINDOW_PX, 0, h - 1))
        x1 = int(np.clip(cx_coarse + REFINE_WINDOW_PX, 0, w - 1))
        y1 = int(np.clip(cy_coarse + REFINE_WINDOW_PX, 0, h - 1))
        max_shift_px = REFINE_MAX_SHIFT_PX

    if (x1 - x0) < 5 or (y1 - y0) < 5:
        return cx_coarse, cy_coarse, "window_too_small"

    crop = I1_raw[y0:y1, x0:x1].astype(np.float32)

    # Smooth
    smoothed  = gaussian(crop, sigma=REFINE_BLUR_SIGMA)
    crop_min  = smoothed.min()
    crop_max  = smoothed.max()
    if crop_max - crop_min < 1e-6:
        return cx_coarse, cy_coarse, "flat_region"
    smoothed_norm = (smoothed - crop_min) / (crop_max - crop_min)

    # Find seed
    cx_local = int(np.clip(cx_coarse - x0, 0, smoothed_norm.shape[1] - 1))
    cy_local = int(np.clip(cy_coarse - y0, 0, smoothed_norm.shape[0] - 1))
    search_r = max(REFINE_WINDOW_PX // 2,
                   int(min(smoothed_norm.shape[:2]) // 4))
    sx0 = max(0, cy_local - search_r)
    sx1 = min(smoothed_norm.shape[0], cy_local + search_r)
    sy0 = max(0, cx_local - search_r)
    sy1 = min(smoothed_norm.shape[1], cx_local + search_r)
    local_patch = smoothed_norm[sx0:sx1, sy0:sy1]
    peak_local  = np.unravel_index(local_patch.argmax(), local_patch.shape)
    seed_y      = sx0 + peak_local[0]
    seed_x      = sy0 + peak_local[1]

    # Otsu + flood-fill
    segment_mask = None
    try:
        otsu_thresh = threshold_otsu(smoothed_norm)
        otsu_mask   = smoothed_norm >= otsu_thresh
        flood_mask  = flood(smoothed_norm, (seed_y, seed_x),
                            tolerance=REFINE_FLOOD_TOLERANCE)
        combined = otsu_mask & flood_mask
        if combined.sum() >= REFINE_MIN_BLOB_PX:
            segment_mask = combined
    except Exception:
        pass

    # Percentile fallback
    if segment_mask is None:
        any_blobs_found  = False
        any_valid_region = False
        for pct in [99, 97, 95, 92, 90]:
            thresh = np.percentile(smoothed_norm, pct)
            bright_mask = smoothed_norm >= thresh
            labeled     = label(bright_mask)
            if labeled.max() == 0:
                continue
            any_blobs_found = True
            best_region = None
            best_dist   = float("inf")
            for region in regionprops(labeled):
                if region.area < REFINE_MIN_BLOB_PX:
                    continue
                ry, rx = region.centroid
                dist = np.sqrt((rx - cx_local)**2 + (ry - cy_local)**2)
                if dist < best_dist:
                    best_dist   = dist
                    best_region = region
            if best_region is None:
                continue
            if best_region.area >= 3:
                any_valid_region = True
                segment_mask     = labeled == best_region.label
                break
        if segment_mask is None:
            if not any_blobs_found:
                return cx_coarse, cy_coarse, "no_blobs_percentile"
            if not any_valid_region:
                return cx_coarse, cy_coarse, "no_valid_blob"

    if segment_mask is None or segment_mask.sum() == 0:
        return cx_coarse, cy_coarse, "no_blobs_otsu"

    ys_px, xs_px     = np.where(segment_mask)
    cx_refined_local = float(np.mean(xs_px))
    cy_refined_local = float(np.mean(ys_px))

    # Sanity check
    shift = np.sqrt((cx_refined_local - cx_local)**2 +
                    (cy_refined_local - cy_local)**2)
    if shift > max_shift_px:
        return cx_coarse, cy_coarse, f"shift_too_large_{shift:.1f}px"

    cx_refined = float(np.clip(cx_refined_local + x0, 0, w - 1))
    cy_refined = float(np.clip(cy_refined_local + y0, 0, h - 1))
    return cx_refined, cy_refined, "ok"


# ─────────────────────────────────────────────
# PER-FILE PROCESSING
# ─────────────────────────────────────────────

def find_png_paths(png_dir, azimuth, mat_stem, angle_idx, mat_path=None):
    base = png_dir if PNG_FLAT_LAYOUT else os.path.join(png_dir, f"{azimuth}Azimuth")
    p_i1 = os.path.join(base, f"{mat_stem}_imagesS{angle_idx}_i1.png")
    p_i2 = os.path.join(base, f"{mat_stem}_imagesS{angle_idx}_i2.png")
    was_gen = False
    if CONVERT_MAT_TO_PNG and mat_path is not None:
        if not (os.path.exists(p_i1) and os.path.exists(p_i2)):
            try:
                p_i1, p_i2 = convert_mat_to_pngs(mat_path, angle_idx, base)
                was_gen = True
            except Exception as e:
                print(f"      [WARN] PNG conversion failed: {e}", flush=True)
    return p_i1, p_i2, was_gen


def _empty_row(azimuth, mat_path, angle_idx):
    return {
        "azimuth":       azimuth,
        "filename":      os.path.basename(mat_path),
        "angle":         angle_idx,
        "cx_coarse":     None, "cy_coarse": None,
        "cx":            None, "cy":        None,
        "yolo_conf":     0.0,
        "yolo_source":   None,
        "refine_reason": "no_yolo_detection",
    }


def process_file(mat_path, png_dir, model, azimuth):
    mat_stem = Path(mat_path).stem
    rows     = []

    for angle_idx in range(ANGLES_PER_FILE):
        p_i1, p_i2, was_gen = find_png_paths(
            png_dir, azimuth, mat_stem, angle_idx, mat_path
        )

        cx_coarse, cy_coarse, conf, source, bbox = get_coarse_centre_from_yolo(
            model, p_i1, p_i2
        )

        if cx_coarse is None:
            rows.append(_empty_row(azimuth, mat_path, angle_idx))
        else:
            try:
                I1_raw = load_raw_I1(mat_path, angle_idx)
                if FLIP_HORIZONTAL:
                    I1_raw = I1_raw[:, ::-1]
                cx_ref, cy_ref, reason = refine_with_brightness_centroid(
                    I1_raw, cx_coarse, cy_coarse, bbox=bbox
                )
            except Exception as e:
                cx_ref, cy_ref, reason = cx_coarse, cy_coarse, "stage2_error"

            shift_px = np.sqrt((cx_ref - cx_coarse)**2 + (cy_ref - cy_coarse)**2)

            if DEBUG:
                _save_debug(mat_path, azimuth, angle_idx,
                            p_i1 if source == "i1" else p_i2,
                            bbox, cx_coarse, cy_coarse,
                            cx_ref, cy_ref, conf, source, reason)

            rows.append({
                "azimuth":       azimuth,
                "filename":      os.path.basename(mat_path),
                "angle":         angle_idx,
                "cx_coarse":     round(cx_coarse, 2),
                "cy_coarse":     round(cy_coarse, 2),
                "cx":            round(cx_ref, 2),
                "cy":            round(cy_ref, 2),
                "yolo_conf":     round(conf, 4),
                "yolo_source":   source,
                "refine_reason": reason,
            })

        if was_gen and not KEEP_PNGS:
            for p in [p_i1, p_i2]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    return rows


# ─────────────────────────────────────────────
# DEBUG OUTPUT SAVER
# ─────────────────────────────────────────────

def _save_debug(mat_path, azimuth, angle_idx, png_path, bbox,
                cx_coarse, cy_coarse, cx_ref, cy_ref, conf, source, reason):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from PIL import Image
    except ImportError:
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)
    stem = Path(mat_path).stem

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if png_path and os.path.exists(png_path):
        axes[0].imshow(np.array(Image.open(png_path)), cmap="gray")
        if bbox is not None:
            bx0, by0, bx1, by1 = bbox
            axes[0].add_patch(mpatches.Rectangle(
                (bx0, by0), bx1 - bx0, by1 - by0,
                linewidth=2, edgecolor="lime", facecolor="none"
            ))
        axes[0].plot(cx_coarse, cy_coarse, "r+", markersize=15, markeredgewidth=2)
        axes[0].plot(cx_ref, cy_ref, "b+", markersize=15, markeredgewidth=2)
        axes[0].set_title(f"YOLO detection ({source})")
    else:
        axes[0].text(0.5, 0.5, "PNG not found", ha="center", va="center")

    try:
        I1_raw = load_raw_I1(mat_path, angle_idx)
        if FLIP_HORIZONTAL:
            I1_raw = I1_raw[:, ::-1]
        axes[1].imshow(I1_raw, cmap="gray")
        win = REFINE_WINDOW_PX
        axes[1].add_patch(mpatches.Rectangle(
            (cx_coarse - win, cy_coarse - win), 2*win, 2*win,
            linewidth=1, edgecolor="yellow", facecolor="none"
        ))
        axes[1].plot(cx_coarse, cy_coarse, "r+", markersize=15, markeredgewidth=2)
        axes[1].plot(cx_ref,    cy_ref,    "b+", markersize=15, markeredgewidth=2)
        axes[1].set_title(f"I1 raw — Centroid Refined ({reason})")
    except Exception:
         pass

    fig.suptitle(f"{stem} | {azimuth}° | Angle {angle_idx}")
    fig.tight_layout()
    out = os.path.join(DEBUG_DIR, f"{stem}_{azimuth}deg_S{angle_idx}.png")
    plt.savefig(out, dpi=100)
    plt.close(fig)


# ─────────────────────────────────────────────
# SEARCH / COLLECTION
# ─────────────────────────────────────────────

def collect_mat_files():
    collected = []
    # Recursively look for any MAT files in visible folder
    for root, dirs, files in os.walk(MAT_DIR):
        for f in sorted(files):
            if f.endswith(".mat"):
                full_path = os.path.join(root, f)
                # Parse azimuth from folder or file
                parent_folder = os.path.basename(root)
                az = None
                if "Azimuth" in parent_folder:
                    try:
                        az = int(parent_folder.replace("Azimuth", ""))
                    except ValueError:
                        pass
                if az is None:
                    # Look in filename e.g. "azimuth_000000"
                    match = re.search(r"azimuth_(\d+)", f, re.IGNORECASE)
                    if match:
                        try:
                            az = int(match.group(1))
                        except ValueError:
                            pass
                if az is None:
                    az = 0
                collected.append((az, full_path))
    return collected


def run():
    print(f"🎬 INITIATING INFERENCE ENGINE", flush=True)
    print(f"Loading weights from: {MODEL_PATH}", flush=True)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model weights best.pt not found at location '{MODEL_PATH}'\n"
              f"Please run the training pipeline first.", flush=True)
        sys.exit(1)

    model = load_yolo_model(MODEL_PATH)
    entries = collect_mat_files()
    if not entries:
        print(f"⚠️ Warning: No .mat dataset files found in {MAT_DIR}", flush=True)
        sys.exit(1)

    if DEBUG and os.path.isdir(DEBUG_DIR):
        try:
            shutil.rmtree(DEBUG_DIR)
        except Exception:
            pass

    print(f"📊 Discovered {len(entries)} .mat files under source layout.", flush=True)

    all_rows = []
    total_start = time.perf_counter()

    for idx, (az, path) in enumerate(entries, 1):
        # Progress signal easily captured by Express child process parser
        print(f"FILE_PROGRESS {idx} {len(entries)} {os.path.basename(path)}", flush=True)
        
        file_start = time.perf_counter()
        try:
            rows = process_file(path, PNG_DIR, model, az)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(path)}: {e}", flush=True)
            for i in range(ANGLES_PER_FILE):
                all_rows.append(_empty_row(az, path, i))

    if not all_rows:
        print("❌ Error: No predictions were successfully generated.", flush=True)
        sys.exit(1)

    # Save outputs CSV
    cols = ["azimuth", "filename", "angle",
            "cx_coarse", "cy_coarse", "cx", "cy",
            "yolo_conf", "yolo_source", "refine_reason"]
    df = pd.DataFrame(all_rows)
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["azimuth", "filename", "angle"]).reset_index(drop=True)
    
    # Ensure directory containing output exists
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)
    
    # Cleanup temp directory
    if os.path.exists(PNG_DIR):
        try:
            shutil.rmtree(PNG_DIR)
        except Exception:
            pass

    total_elapsed = time.perf_counter() - total_start
    print(f"🎉 Inference pipeline completed successfully in {total_elapsed:.2f} seconds.", flush=True)
    print(f"📊 Exported results to: {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    run()
