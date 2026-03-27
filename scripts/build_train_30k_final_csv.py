# scripts/build_train_30k_final_csv.py
import os
import re
import pandas as pd

SPLIT_TXT = "data/splits/train_30k_files.txt"
IN_CSV = "data/labels/train_30k.csv"
OUT_CSV = "data/labels/train_30k_final.csv"
PROJECT_ROOT = "/home/azureuser/chexpert"
OUT_IMAGES_DIR = "data/images_30k_final"

def full_path(p: str) -> str:
    p = p.strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p
    candidate = os.path.join(PROJECT_ROOT, p)
    return candidate

def dest_name_from_src(src: str) -> str:
    patient = re.findall(r"(patient\d+)", src)
    study = re.findall(r"(study\d+)", src)
    view = os.path.basename(src)  # view1_frontal.jpg, view2_lateral.jpg, etc.
    if not patient or not study:
        # fallback (shouldn't happen for CheXpert paths)
        return view
    return f"{patient[-1]}_{study[-1]}_{view}"

def main():
    df = pd.read_csv(IN_CSV)

    # find image column
    img_col = None
    for c in ["Path", "path", "image", "Image", "filename", "Filename"]:
        if c in df.columns:
            img_col = c
            break
    if img_col is None:
        raise RuntimeError(f"No image column found in {IN_CSV}. Columns: {list(df.columns)}")

    with open(SPLIT_TXT, "r") as f:
        paths = [line.strip() for line in f if line.strip()]

    if len(paths) != len(df):
        raise RuntimeError(f"Row mismatch: split has {len(paths)} lines, csv has {len(df)} rows")

    # Replace image column using split list order (this is the source of truth)
    new_names = []
    missing_src = 0
    for p in paths:
        src = full_path(p)
        if not os.path.exists(src):
            missing_src += 1
        new_names.append(dest_name_from_src(src))

    if missing_src:
        print("WARNING: missing src files:", missing_src)

    df[img_col] = new_names
    df.to_csv(OUT_CSV, index=False)

    print("Wrote:", OUT_CSV)
    print("Image column:", img_col)
    print("Example:", df[img_col].iloc[0])
    print("Last:", df[img_col].iloc[-1])

    # Verify names exist in the destination folder
    names = set(os.listdir(OUT_IMAGES_DIR))
    miss = sum(1 for x in df[img_col].head(500) if str(x) not in names)
    print("Check first 500 rows missing:", miss)

if __name__ == "__main__":
    main()
