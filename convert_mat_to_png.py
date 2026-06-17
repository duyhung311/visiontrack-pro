import os
import sys
import argparse
import glob
import numpy as np
import scipy.io as sio
from pathlib import Path
from skimage import img_as_float
from skimage.exposure import rescale_intensity
import cv2

def load_raw_I2(mat_path: str, angle_idx: int) -> np.ndarray:
    mat       = sio.loadmat(mat_path)
    imdat     = mat["imdat"]
    angle_key = f"imagesS{angle_idx}"
    I2_raw    = imdat[angle_key][0][0]["presetcapture"][0][0]["image"][0][1]
    return rescale_intensity(
        img_as_float(np.squeeze(I2_raw)), out_range=(0.0, 1.0)
    ).astype(np.float32)

def convert_mat_to_pngs(mat_path: str, angle_idx: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    mat_stem = Path(mat_path).stem

    I2 = load_raw_I2(mat_path, angle_idx)

    path_i2 = os.path.join(out_dir, f"{mat_stem}_imagesS{angle_idx}_i2.png")
    cv2.imwrite(path_i2, (I2 * 255.0).astype(np.uint8))
    return path_i2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="", help="Subfolder name inside mat_files / images")
    args = parser.parse_args()

    project_root = os.getcwd()
    visible_dir = os.path.join(project_root, "visible")
    
    # Resolve directories
    if args.folder and args.folder not in ["Default Workspace", "Default (visible)", "default"]:
        mat_src_dir = os.path.join(visible_dir, "mat_files", args.folder)
        img_dest_dir = os.path.join(visible_dir, "images", args.folder)
    else:
        mat_src_dir = os.path.join(visible_dir, "mat_files")
        img_dest_dir = os.path.join(visible_dir, "images")

    print(f"Source MAT directory: {mat_src_dir}", flush=True)
    print(f"Destination PNG directory: {img_dest_dir}", flush=True)

    if not os.path.exists(mat_src_dir):
        print(f"❌ Error: Source directory does not exist: {mat_src_dir}", flush=True)
        sys.exit(1)

    mat_files = glob.glob(os.path.join(mat_src_dir, "*.mat"))
    print(f"Found {len(mat_files)} .mat files to convert.", flush=True)

    if not mat_files:
        print("⚠️ No .mat files found in current folder workspace.", flush=True)
        sys.exit(0)

    os.makedirs(img_dest_dir, exist_ok=True)

    success_count = 0
    total_images_written = 0

    for idx, mat_path in enumerate(mat_files, 1):
        filename = os.path.basename(mat_path)
        print(f"CONVERSION_PROGRESS {idx} {len(mat_files)} {filename}", flush=True)
        try:
            for angle_idx in range(5):
                p_i2 = convert_mat_to_pngs(mat_path, angle_idx, img_dest_dir)
                total_images_written += 1
            success_count += 1
            print(f"Converted {filename} successfully.", flush=True)
        except Exception as e:
            print(f"Error converting {filename}: {e}", flush=True)

    print(f"🎉 Conversion finalized. Converted {success_count}/{len(mat_files)} .mat files. Generated {total_images_written} PNG images.", flush=True)

if __name__ == "__main__":
    main()
