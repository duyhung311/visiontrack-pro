# Pseudopupil Detection — YOLO OBB Training Configuration
# Runs training asynchronously with live progress callbacks to Express backend.

import os
import json
import shutil
import yaml
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import cv2
from pathlib import Path
from collections import defaultdict
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from scipy.ndimage import uniform_filter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION & PARAMETERS
# ─────────────────────────────────────────────

SCIENTIFIC_NAME  = "Aeschna_isoceles"
# By default, files are inside flat "visible" directory or with subfolders
MAT_DIR          = "./visible/mat_files"
# Label JSON file
LABEL_STUDIO_JSON = "./visible/labels.json"

# Fallback if no specific JSON file is in ./visible/labels or ./visible
# Search directory for any .json file
def find_label_json(data_dir=None):
    search_dirs = []
    if data_dir:
        # If data_dir has "mat_files" in path, replace it with "labels" as prime target
        if "mat_files" in data_dir:
            search_dirs.append(data_dir.replace("mat_files", "labels"))
        search_dirs.append(os.path.join(data_dir, "labels"))
        search_dirs.append(data_dir)
    search_dirs.extend(["./visible/labels", "./visible", "."])
    for base in search_dirs:
        if os.path.exists(base) and os.path.isdir(base):
            for f in sorted(os.listdir(base)):
                if f.endswith(".json"):
                    return os.path.join(base, f)
    return "./project-11-at-2026-05-27-05-46-73bcca98.json" # fallback target from template

# Where to write the YOLO dataset (images + labels + yaml)
DATASET_DIR      = "./yolo_dataset"

# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train YOLO OBB model for pseudopupil detection.")
parser.add_argument("--version", type=str, default="8", help="YOLO version (8, 11, 26 only)")
parser.add_argument("--size", type=str, default="m", help="YOLO size (n, s, m, l only)")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs") # Default to low count for fast demo runs, but support high count
parser.add_argument("--batch", type=int, default=16, help="Batch size")
parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, or GPU index)")
parser.add_argument("--label_file", type=str, default="", help="Label Studio JSON file path")
parser.add_argument("--data_dir", type=str, default="", help="Custom dataset base path under visible")
args = parser.parse_args()

YOLO_VERSION = args.version
YOLO_SIZE = args.size
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch
DEVICE = args.device

if args.data_dir:
    if "mat_files" in args.data_dir:
        MAT_DIR = args.data_dir
    else:
        MAT_DIR = os.path.join(args.data_dir, "mat_files")

if args.label_file:
    LABEL_STUDIO_JSON = args.label_file
else:
    LABEL_STUDIO_JSON = find_label_json(args.data_dir)

# Base model:
#   v8 OBB: yolov8{size}-obb.pt
#   v11 OBB: yolo11{size}-obb.pt
#   v26 fallback: yolov26{size}-obb.pt or yolo26{size}-obb.pt
if YOLO_VERSION == "8":
    YOLO_BASE_MODEL = f"yolov8{YOLO_SIZE}-obb.pt"
elif YOLO_VERSION == "11":
    YOLO_BASE_MODEL = f"yolo11{YOLO_SIZE}-obb.pt"
else:
    YOLO_BASE_MODEL = f"yolov{YOLO_VERSION}{YOLO_SIZE}-obb.pt"

# Output weights location after training
RUNS_DIR         = "./runs"
RUN_NAME         = f"pseudopupil_yolo{YOLO_VERSION}{YOLO_SIZE}_obb"

IMAGE_SIZE       = 640
LR0              = 0.01
LRF              = 0.01
WEIGHT_DECAY     = 5e-4
WARMUP_EPOCHS    = 3
PATIENCE         = 15
NUM_WORKERS      = 4
SEED             = 42
VAL_SPLIT        = 0.15

print(f"🚀 INITIALIZING TRAINING PIPELINE", flush=True)
print(f"YOLO Configuration: Version={YOLO_VERSION}, Size={YOLO_SIZE}", flush=True)
print(f"Base Model Target:  {YOLO_BASE_MODEL}", flush=True)
print(f"Dataset root:       {MAT_DIR}", flush=True)
print(f"Label Studio JSON:  {LABEL_STUDIO_JSON}", flush=True)
print(f"Hyperparameters:    Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, Device={DEVICE}", flush=True)

# ─────────────────────────────────────────────
# LABEL STUDIO JSON PARSING  (i1 only)
# ─────────────────────────────────────────────

def parse_label_studio_json(json_path: str) -> list:
    if not os.path.exists(json_path):
        print(f"❌ Error: Label JSON file not found at {json_path}", flush=True)
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    records = []
    skipped_i2    = 0
    skipped_empty = 0

    for entry in data:
        img_path = entry.get("image", "")
        # Grab name from path or field
        fname = img_path.split("/")[-1]

        # Filter: check if "_i1" is in fname or it ends with "_i1.png"
        fname_lower = fname.lower()
        if "_i1.png" not in fname_lower and "_i1" not in fname_lower:
            skipped_i2 += 1
            continue

        # Skip entries with no annotation
        labels = entry.get("label", [])
        if not labels:
            skipped_empty += 1
            continue

        # Parse filename → mat_stem + angle_key
        mat_stem = None
        angle_key = None

        # 1. Regex approach (robust fallback)
        import re
        match = re.search(r"(imagesS\d+)_(i1|i2|both)\.?([a-zA-Z0-9]+)?\.png$", fname, re.IGNORECASE)
        if match:
            angle_key = match.group(1)
            end_idx = fname.lower().find(angle_key.lower())
            if end_idx != -1:
                mat_stem = fname[:end_idx].rstrip("-_")

        # 2. Legacy fallback
        if not mat_stem or not angle_key:
            stem_no_ext = fname.replace(".png", "")
            parts       = stem_no_ext.rsplit("_", 2)
            if len(parts) == 3 and parts[1].startswith("imagesS"):
                mat_stem  = parts[0]
                angle_key = parts[1]

        if not mat_stem or not angle_key:
            continue

        # Parse bbox (percentage → normalised 0-1)
        box       = labels[0]
        orig_w    = int(box["original_width"])
        orig_h    = int(box["original_height"])

        x_pct  = box["x"]
        y_pct  = box["y"]
        w_pct  = box["width"]
        h_pct  = box["height"]

        # Convert to normalised centre coordinates
        xc_n = float(np.clip((x_pct + w_pct / 2.0) / 100.0, 0.0, 1.0))
        yc_n = float(np.clip((y_pct + h_pct / 2.0) / 100.0, 0.0, 1.0))
        bw_n = float(np.clip(w_pct / 100.0, 1e-4, 1.0))
        bh_n = float(np.clip(h_pct / 100.0, 1e-4, 1.0))

        records.append({
            "mat_stem":  mat_stem,
            "angle_key": angle_key,
            "png_name":  fname,
            "xc_n":      xc_n,
            "yc_n":      yc_n,
            "bw_n":      bw_n,
            "bh_n":      bh_n,
            "orig_w":    orig_w,
            "orig_h":    orig_h,
        })

    print(f"📊 Parsed Label Studio JSON: {len(data)} total entries", flush=True)
    print(f"   Stored {len(records)} valid i1 anchors", flush=True)
    print(f"   Ignored {skipped_i2} i2 images | Skipped {skipped_empty} empty annotations", flush=True)
    return records


# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────

def split_by_file(records, val_fraction=0.15, seed=42):
    rng      = np.random.default_rng(seed)
    by_stem  = defaultdict(list)
    for i, r in enumerate(records):
        by_stem[r["mat_stem"]].append(i)
    stems    = list(by_stem.keys())
    rng.shuffle(stems)
    n_val    = max(1, int(len(stems) * val_fraction))
    val_stems = set(stems[:n_val])
    train_records = [records[i] for s in stems if s not in val_stems for i in by_stem[s]]
    val_records   = [records[i] for s in val_stems for i in by_stem[s]]
    return train_records, val_records


# ─────────────────────────────────────────────
# IMAGE LOADING + CHANNEL BUILDING
# ─────────────────────────────────────────────

def load_raw_images(mat_path: str, angle_key: str):
    mat    = sio.loadmat(mat_path)
    imdat  = mat["imdat"]
    I1_raw = imdat[angle_key][0][0]["presetcapture"][0][0]["image"][0][0]
    I2_raw = imdat[angle_key][0][0]["presetcapture"][0][0]["image"][0][1]
    I1 = rescale_intensity(
        img_as_float(np.squeeze(I1_raw)), out_range=(0.0, 1.0)
    ).astype(np.float32)
    I2 = rescale_intensity(
        img_as_float(np.squeeze(I2_raw)), out_range=(0.0, 1.0)
    ).astype(np.float32)
    return I1, I2


def local_contrast_norm(img: np.ndarray, sigma: int = 32) -> np.ndarray:
    local_mean = uniform_filter(img.astype(np.float32), size=sigma)
    gate   = (local_mean > 0.02).astype(np.float32)
    normed = (img / (local_mean + 0.02)) * gate
    return rescale_intensity(normed, out_range=(0.0, 1.0)).astype(np.float32)


def build_channels(I1: np.ndarray, I2: np.ndarray,
                   exposure: str = "both") -> np.ndarray:
    if exposure == "i1":
        c = local_contrast_norm(I1)
        return np.stack([c, c, np.zeros_like(c)], axis=-1)
    elif exposure == "i2":
        c = local_contrast_norm(I2)
        return np.stack([c, c, np.zeros_like(c)], axis=-1)
    else:  # "both"
        diff = np.abs(I2 - I1).astype(np.float32)
        return np.stack([
            local_contrast_norm(I1),
            local_contrast_norm(I2),
            local_contrast_norm(diff),
        ], axis=-1)


def channels_to_uint8(img_hwc: np.ndarray) -> np.ndarray:
    return (np.clip(img_hwc, 0.0, 1.0) * 255.0).astype(np.uint8)


# ─────────────────────────────────────────────
# DATASET EXPORT
# ─────────────────────────────────────────────

def find_mat_path(mat_dir: str, mat_stem: str) -> str:
    # 1. Exact case-sensitive match
    direct = os.path.join(mat_dir, f"{mat_stem}.mat")
    if os.path.exists(direct):
         return direct

    # 2. Case-insensitive exact match
    if os.path.exists(mat_dir):
        mat_stem_lower = mat_stem.lower()
        for f in os.listdir(mat_dir):
            if f.lower() == f"{mat_stem_lower}.mat":
                return os.path.join(mat_dir, f)

    # 3. Handle Label Studio prepended hashes/hyphens (e.g., "cf0927df-Aesch")
    if "-" in mat_stem:
        parts = mat_stem.split("-")
        for i in range(1, len(parts) + 1):
            sub_stem = "-".join(parts[-i:])
            p = os.path.join(mat_dir, f"{sub_stem}.mat")
            if os.path.exists(p):
                return p
            # Try case-insensitive
            if os.path.exists(mat_dir):
                for f in os.listdir(mat_dir):
                    if f.lower() == f"{sub_stem.lower()}.mat":
                        return os.path.join(mat_dir, f)

    # 4. Loose substring match as fallback
    if os.path.exists(mat_dir):
        mat_stem_clean = mat_stem.lower().replace("_", "").replace("-", "")
        # Remove common hex prefix if any
        import re
        pref_match = re.match(r"^[a-f0-9]{8}-", mat_stem.lower())
        if pref_match:
            stripped_stem = mat_stem.lower()[9:].replace("_", "").replace("-", "")
        else:
            stripped_stem = mat_stem_clean

        best_match = None
        for f in sorted(os.listdir(mat_dir)):
            if f.endswith(".mat"):
                f_stem = f[:-4]
                f_stem_clean = f_stem.lower().replace("_", "").replace("-", "")
                if f_stem_clean == stripped_stem or f_stem_clean == mat_stem_clean:
                    return os.path.join(mat_dir, f)
                if f_stem_clean in stripped_stem or stripped_stem in f_stem_clean:
                    best_match = os.path.join(mat_dir, f)
        if best_match:
            return best_match

    return None


def export_dataset(records, mat_dir, dataset_dir, split="train"):
    img_dir = os.path.join(dataset_dir, "images", split)
    lbl_dir = os.path.join(dataset_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    exported  = []
    not_found = 0

    for idx, r in enumerate(records):
        mat_path = find_mat_path(mat_dir, r["mat_stem"])
        if mat_path is None:
            not_found += 1
            continue

        try:
            I1, I2 = load_raw_images(mat_path, r["angle_key"])
        except Exception as e:
            continue

        # Normalised top-left, top-right, bottom-right, bottom-left clockwise corners
        xc, yc = r["xc_n"], r["yc_n"]
        hw, hh = r["bw_n"] / 2.0, r["bh_n"] / 2.0
        x1, y1 = xc - hw, yc - hh
        x2, y2 = xc + hw, yc - hh
        x3, y3 = xc + hw, yc + hh
        x4, y4 = xc - hw, yc + hh
        label_line = (f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
                      f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n")

        base_stem = r["png_name"].replace(".png", "").replace("_i1", "")

        for exposure in ("i1", "i2", "both"):
            try:
                channels  = build_channels(I1, I2, exposure)
                img_u8    = channels_to_uint8(channels)
                stem      = f"{base_stem}_{exposure}"
                png_path  = os.path.join(img_dir, f"{stem}.png")
                txt_path  = os.path.join(lbl_dir, f"{stem}.txt")
                cv2.imwrite(png_path, img_u8)
                with open(txt_path, "w") as f:
                    f.write(label_line)
                exported.append((png_path, r["xc_n"], r["yc_n"],
                                 r["orig_w"], r["orig_h"]))
            except Exception as e:
                pass

        # Periodically emit progress during export
        if idx % 10 == 0 or idx == len(records) - 1:
            raw_pct = int(((idx + 1) / len(records)) * 100)
            print(f"📦 Channel building export ({split}): {raw_pct}%", flush=True)

    n_labels = len(records) - not_found
    print(f"✅ [{split}] Staging complete: {n_labels} files -> {len(exported)} images", flush=True)
    return exported


def write_dataset_yaml(dataset_dir: str) -> str:
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    cfg = {
        "path":  os.path.abspath(dataset_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["pseudopupil"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return yaml_path


# ─────────────────────────────────────────────
# POST-TRAINING DIAGNOSTIC
# ─────────────────────────────────────────────

def run_val_diagnostic(model_path: str, val_exported: list):
    from ultralytics import YOLO
    print("\n🔍 Running validation-set diagnostic error check...", flush=True)
    try:
        model  = YOLO(model_path)
    except Exception as e:
        print(f"⚠️ Failed to load best weights for diagnostic: {e}", flush=True)
        return

    errors = []

    for img_path, xc_n_gt, yc_n_gt, orig_w, orig_h in val_exported:
        if not os.path.exists(img_path):
            continue

        cx_gt = xc_n_gt * orig_w
        cy_gt = yc_n_gt * orig_h

        results = model.predict(source=img_path, verbose=False, conf=0.01)
        if not results or results[0].obb is None or len(results[0].obb) == 0:
            continue

        xywhr = results[0].obb.xywhr.cpu().numpy()
        confs  = results[0].obb.conf.cpu().numpy()
        best   = int(np.argmax(confs))
        cx_pred = float(xywhr[best, 0])
        cy_pred = float(xywhr[best, 1])

        errors.append({
            "err_x": cx_pred - cx_gt,
            "err_y": cy_pred - cy_gt,
            "dist":  np.sqrt((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2),
        })

    if not errors:
        print("💡 Diagnostic: No detections on holdout validation subset", flush=True)
        return

    df = pd.DataFrame(errors)
    med_err = df.dist.median()
    under10 = (df.dist < 10).mean()*100
    under20 = (df.dist < 20).mean()*100
    print(f"🎯 Validation Diagnostic Results:", flush=True)
    print(f"   Median YOLO box-centre error : {med_err:.2f} px", flush=True)
    print(f"   Detections within 10px       : {under10:.1f}%", flush=True)
    print(f"   Detections within 20px       : {under20:.1f}%", flush=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def train():
    # 1. Parse JSON
    records = parse_label_studio_json(LABEL_STUDIO_JSON)
    if not records:
        print("❌ Error: No valid annotations imported. Check label contents.", flush=True)
        sys.exit(1)

    # 2. Train/val split
    train_records, val_records = split_by_file(records, VAL_SPLIT, SEED)
    print(f"📈 Split structured: Train={len(train_records)} files | Heldout Val={len(val_records)} files", flush=True)

    # 3. Export images
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    train_exported = export_dataset(train_records, MAT_DIR, DATASET_DIR, "train")
    val_exported   = export_dataset(val_records,   MAT_DIR, DATASET_DIR, "val")

    # 4. Write dataset yaml
    yaml_path = write_dataset_yaml(DATASET_DIR)

    # 5. Bring in YOLO and register local stdout callback
    print(f"💾 Loading ultralytics model...", flush=True)
    from ultralytics import YOLO
    
    # Check if we should override run folder to ensure exactly 1 configuration file
    # Clear preexisting weights in RUNS_DIR/RUN_NAME before training to avoid stacking directory suffixes
    target_run_dir = os.path.join(RUNS_DIR, RUN_NAME)
    if os.path.exists(target_run_dir):
        print(f"🧹 Clearing previous model weights at {target_run_dir} to overwrite clean...", flush=True)
        # Safely remove it
        try:
            shutil.rmtree(target_run_dir)
        except Exception as e:
            print(f"⚠️ Warning during clean: {e}", flush=True)

    model = YOLO(YOLO_BASE_MODEL)

    # Callback to feed progress to express server
    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        total = trainer.epochs
        loss_str = ""
        loss_items = getattr(trainer, "loss_items", None)
        if loss_items is not None:
            # Try to map names if available
            loss_names = getattr(trainer, "loss_names", getattr(trainer, "loss_keys", None))
            vals = []
            if hasattr(loss_items, "items"):
                try:
                    loss_str = " ".join([f"{k}:{float(v):.4f}" for k, v in loss_items.items()])
                except Exception:
                    pass
            elif hasattr(loss_items, "tolist"):
                try:
                    val_data = loss_items.tolist()
                    if isinstance(val_data, list):
                        vals = val_data
                    else:
                        vals = [val_data]
                except Exception:
                    pass
            elif isinstance(loss_items, (list, tuple)):
                vals = list(loss_items)
            else:
                try:
                    vals = [float(loss_items)]
                except Exception:
                    pass

            if not loss_str and vals:
                if loss_names and len(loss_names) == len(vals):
                    try:
                        loss_str = " ".join([f"{k}:{float(v):.4f}" for k, v in zip(loss_names, vals)])
                    except Exception:
                        pass
                else:
                    try:
                        loss_str = " ".join([f"loss_{i}:{float(v):.4f}" for i, v in enumerate(vals)])
                    except Exception:
                        pass
        # Format easily readable by Node server child parser
        print(f"EPOCH_PROGRESS {epoch} {total} {loss_str}", flush=True)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    print(f"🔥 Training starting...", flush=True)
    model.train(
        data          = yaml_path,
        epochs        = NUM_EPOCHS,
        imgsz         = IMAGE_SIZE,
        batch         = BATCH_SIZE,
        device        = DEVICE,
        workers       = NUM_WORKERS,
        seed          = SEED,
        lr0           = LR0,
        lrf           = LRF,
        weight_decay  = WEIGHT_DECAY,
        warmup_epochs = WARMUP_EPOCHS,
        patience      = PATIENCE,
        project       = RUNS_DIR,
        name          = RUN_NAME,
        exist_ok      = True,

        # Augmentation
        fliplr    = 0.5,
        flipud    = 0.2,
        degrees   = 45.0,
        translate = 0.2,
        scale     = 0.3,
        mosaic    = 0.5,
        mixup     = 0.1,
        hsv_h     = 0.0,
        hsv_s     = 0.0,
        hsv_v     = 0.4,
        verbose   = True,
    )

    best_pt = os.path.join(RUNS_DIR, RUN_NAME, "weights", "best.pt")
    if os.path.exists(best_pt):
        print(f"✅ Success! Training complete.", flush=True)
        print(f"💾 Saved best weights to: {best_pt}", flush=True)
        # Execute diagnostic
        run_val_diagnostic(best_pt, val_exported)
    else:
        print(f"⚠️ Warning: best.pt weights targets missing at {best_pt}", flush=True)

    # 8. Clean up dataset files to save storage space
    print("🧹 Cleaning up dataset files...", flush=True)
    for split in ("train", "val"):
        for subf in ("images", "labels"):
            folder = os.path.join(DATASET_DIR, subf, split)
            if os.path.isdir(folder):
                shutil.rmtree(folder)
    print("✨ Process completed successfully.", flush=True)

if __name__ == "__main__":
    train()
