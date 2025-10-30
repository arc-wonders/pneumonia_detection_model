import os, shutil, random, re
from pathlib import Path
from typing import List, Dict, Tuple

# ---------------- CONFIG ----------------
SRC = Path(r"D:\pneumonia_detection_model\chest-xray-pneumonia\versions\2\chest_xray")
DST = Path(r"D:\pneumonia_detection_model\dataset")  # new clean dataset
VAL_FRAC = 0.15          # 15% of TRAIN per class goes to VAL
SEED = 42                # reproducibility
COPY_INSTEAD_OF_MOVE = True  # keep original intact
# ----------------------------------------

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]

random.seed(SEED)

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VALID_EXT and p.stat().st_size > 0

def patient_id_from_name(name: str) -> str | None:
    # Try to capture 'person123_' pattern (common in this dataset for PNEUMONIA)
    m = re.search(r"(person\d+)_", name.lower())
    return m.group(1) if m else None

def list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists(): return []
    return [p for p in dir_path.iterdir() if is_image_file(p)]

def ensure_dirs(base: Path):
    for s in SPLITS:
        for c in CLASSES:
            (base / s / c).mkdir(parents=True, exist_ok=True)

def cp_or_mv(src: Path, dst: Path):
    if COPY_INSTEAD_OF_MOVE:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)

def copy_tree(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in list_images(src_dir):
        cp_or_mv(p, dst_dir / p.name)

def recount(base: Path) -> Dict[str, Dict[str, int]]:
    counts = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for s in SPLITS:
        for c in CLASSES:
            d = base / s / c
            if d.exists():
                counts[s][c] = len(list_images(d))
    return counts

def print_counts(title: str, counts: Dict[str, Dict[str, int]]):
    print(f"\n=== {title} ===")
    grand = 0
    for s in SPLITS:
        row = sum(counts[s][c] for c in CLASSES)
        grand += row
        print(f"{s:>5}: " + "  ".join(f"{c}={counts[s][c]}" for c in CLASSES) + f"   (total {row})")
    print(f"Grand total: {grand}")

def stratified_patient_split(train_cls_files: List[Path], val_frac: float) -> Tuple[List[Path], List[Path]]:
    # Group by patient if possible
    groups: Dict[str, List[Path]] = {}
    files_wo_patient: List[Path] = []
    for p in train_cls_files:
        pid = patient_id_from_name(p.name)
        if pid: groups.setdefault(pid, []).append(p)
        else: files_wo_patient.append(p)

    # If we have enough patient groups, sample by groups; else fallback to file-level
    if len(groups) >= 5:
        pids = list(groups.keys())
        random.shuffle(pids)
        # Greedy add groups until we reach target size
        total = sum(len(v) for v in groups.values()) + len(files_wo_patient)
        target = max(1, int(round(val_frac * total)))
        val_files: List[Path] = []
        picked = set()
        for pid in pids:
            if len(val_files) >= target: break
            val_files.extend(groups[pid])
            picked.add(pid)
        # If still short, fill with individual files lacking patient id
        if len(val_files) < target:
            random.shuffle(files_wo_patient)
            need = target - len(val_files)
            val_files.extend(files_wo_patient[:need])
            train_rest = files_wo_patient[need:]
        else:
            train_rest = files_wo_patient[:]
        # Remaining groups go to train
        train_files: List[Path] = []
        for pid, imgs in groups.items():
            if pid not in picked:
                train_files.extend(imgs)
        train_files.extend(train_rest)
        return train_files, val_files
    else:
        # Fallback: file-level split
        files = train_cls_files[:]
        random.shuffle(files)
        k = max(1, int(round(val_frac * len(files))))
        val_files = files[:k]
        train_files = files[k:]
        return train_files, val_files

def main():
    # Basic checks
    if not (SRC / "train").exists() or not (SRC / "test").exists():
        print("ERROR: SRC does not look like the dataset root with train/test folders.")
        print(f"SRC = {SRC}")
        return

    # Start fresh destination
    if DST.exists():
        print(f"Destination exists: {DST}")
        print("Will add/overwrite files if name-collisions occur. (Keeps originals intact if copying.)")
    ensure_dirs(DST)

    # 1) Copy TEST exactly as-is
    for c in CLASSES:
        copy_tree(SRC / "test" / c, DST / "test" / c)

    # 2) Build new TRAIN/VAL from SRC/train
    for c in CLASSES:
        train_cls_files = list_images(SRC / "train" / c)
        train_keep, val_take = stratified_patient_split(train_cls_files, VAL_FRAC)

        # Copy TRAIN keepers
        for p in train_keep:
            cp_or_mv(p, (DST / "train" / c) / p.name)

        # Copy VAL
        for p in val_take:
            cp_or_mv(p, (DST / "val" / c) / p.name)

    # 3) Report counts
    src_counts = {
        "train": {cls: len(list_images(SRC / "train" / cls)) for cls in CLASSES},
        "val":   {cls: len(list_images(SRC / "val" / cls))   for cls in CLASSES},
        "test":  {cls: len(list_images(SRC / "test" / cls))  for cls in CLASSES},
    }
    print_counts("Original (for reference, may be unchanged)", src_counts)

    dst_counts = recount(DST)
    print_counts("New dataset (DST)", dst_counts)

    print(f"\n✅ Done. New clean dataset at: {DST}")
    print("Point your code to DATA_ROOT = r\"" + str(DST) + "\"")

if __name__ == "__main__":
    main()
