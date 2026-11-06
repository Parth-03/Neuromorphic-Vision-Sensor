import os, zipfile, shutil
from pathlib import Path, PurePosixPath
import pandas as pd

# =========================
# USER SETTINGS (edit these)
# =========================

ZIP_PATH = r"D:\Neuromorph data\20bn-jester-event.zip"
TRAIN_CSV = r"D:\Neuromorph data\20bn-jester-event\Train.csv"
VAL_CSV = r"D:\Neuromorph data\20bn-jester-event\Validation.csv"
OUT_DIR = r"D:\Neuromorph data\20bn-jester-event\Subset"

# Your chosen 5 classes (must match CSV label strings exactly)
TARGET_LABELS = [
    "Thumb Up",
    "Shaking Hand",
    "Swiping Up",
    "Zooming In With Full Hand",
    "Turning Hand Clockwise",
]

TRAIN_PER_CLASS = 300
VAL_PER_CLASS   = 30
TEST_PER_CLASS  = 30
SEED            = 0
MAKE_ZIP        = True

# Inside the big zip:
ZIP_ROOT = "Jester_Event"
ZIP_SPLIT_TRAIN = "Train"
ZIP_SPLIT_VAL   = "Validation"

# How we want it laid out in the subset folder:
OUT_SPLIT_TRAIN = "train"
OUT_SPLIT_VAL   = "validation"


# =========================
# Helpers
# =========================
def balanced_sample(df: pd.DataFrame, labels, n_per_class, seed=0):
    parts = []
    for lab in labels:
        sub = df[df["label"] == lab]
        if len(sub) < n_per_class:
            raise ValueError(f"Not enough samples for '{lab}': have {len(sub)}, need {n_per_class}")
        parts.append(sub.sample(n=n_per_class, random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def build_splits(train_df, val_df, seed=0, train_per=300, val_per=30, test_per=30):
    train_small = balanced_sample(train_df, TARGET_LABELS, n_per_class=train_per, seed=seed)

    need = val_per + test_per  # 60 per class from Validation.csv
    val60 = balanced_sample(val_df, TARGET_LABELS, n_per_class=need, seed=seed + 1)

    val_parts, test_parts = [], []
    for lab in TARGET_LABELS:
        sub = val60[val60["label"] == lab].reset_index(drop=True)
        val_parts.append(sub.iloc[:val_per])
        test_parts.append(sub.iloc[val_per:val_per + test_per])

    val_small  = pd.concat(val_parts).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    test_small = pd.concat(test_parts).sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)

    label_to_idx = {lab:i for i, lab in enumerate(TARGET_LABELS)}
    for df_ in (train_small, val_small, test_small):
        df_["y"] = df_["label"].map(label_to_idx).astype(int)

    return train_small, val_small, test_small

def extract_from_zip(zip_path, out_root, zip_root, zip_split, out_split, needed_video_ids):
    """
    Extract:
      {zip_root}/{zip_split}/{video_id}/events.h5
      {zip_root}/{zip_split}/{video_id}/v2e-script-args.txt
    into:
      {out_root}/{out_split}/{video_id}/...
    """
    out_root = Path(out_root)
    needed = set(str(int(v)) for v in needed_video_ids)

    extracted_files = 0
    missing_ids = set(needed)

    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue

            p = PurePosixPath(info.filename)
            parts = p.parts

            # Must match expected hierarchy exactly:
            # Jester_Event / Train(or Validation) / <vid> / <file>
            if len(parts) < 4:
                continue
            if parts[0] != zip_root:
                continue
            if parts[1] != zip_split:
                continue

            vid = parts[2]
            if vid not in needed:
                continue

            fname = parts[-1]
            if fname not in ("events.h5", "v2e-script-args.txt"):
                continue

            target = out_root / out_split / vid / fname
            target.parent.mkdir(parents=True, exist_ok=True)

            with z.open(info, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted_files += 1
            # mark id as seen (may still be missing one of the two files; we'll check later)
            if vid in missing_ids:
                # don't remove yet; we remove only after checking both exist
                pass

    # After extraction, compute which video folders are complete
    actually_missing = []
    for vid in needed:
        folder = out_root / out_split / vid
        if not (folder / "events.h5").exists() or not (folder / "v2e-script-args.txt").exists():
            actually_missing.append(vid)

    print(f"[{zip_split}] extracted files: {extracted_files}")
    if actually_missing:
        print(f"[{zip_split}] WARNING: {len(actually_missing)} video_ids missing one/both files. Example:", actually_missing[:10])

def zip_subset_folder(folder_path):
    folder_path = Path(folder_path)
    zip_path = folder_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder_path.rglob("*"):
            if p.is_file():
                arc = Path(folder_path.name) / p.relative_to(folder_path)
                z.write(p, arcname=str(arc))
    print("Wrote:", zip_path)
    return zip_path


# =========================
# Main
# =========================
def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    # quick sanity: ensure labels exist
    for lab in TARGET_LABELS:
        if lab not in set(train_df["label"].unique()):
            raise ValueError(f"Label '{lab}' not found in Train.csv. Check spelling/casing.")
        if lab not in set(val_df["label"].unique()):
            raise ValueError(f"Label '{lab}' not found in Validation.csv. Check spelling/casing.")

    train_small, val_small, test_small = build_splits(
        train_df, val_df,
        seed=SEED,
        train_per=TRAIN_PER_CLASS,
        val_per=VAL_PER_CLASS,
        test_per=TEST_PER_CLASS
    )

    # Save CSVs
    train_small.to_csv(out / "Train_small.csv", index=False)
    val_small.to_csv(out / "Validation_small.csv", index=False)
    test_small.to_csv(out / "Test_small.csv", index=False)

    train_ids = train_small["video_id"].astype(int).tolist()
    val_ids   = val_small["video_id"].astype(int).tolist()
    test_ids  = test_small["video_id"].astype(int).tolist()

    # Extract from ZIP (note: test is coming from Validation split)
    extract_from_zip(ZIP_PATH, out, ZIP_ROOT, ZIP_SPLIT_TRAIN, OUT_SPLIT_TRAIN, train_ids)
    extract_from_zip(ZIP_PATH, out, ZIP_ROOT, ZIP_SPLIT_VAL,   OUT_SPLIT_VAL,   val_ids)
    extract_from_zip(ZIP_PATH, out, ZIP_ROOT, ZIP_SPLIT_VAL,   OUT_SPLIT_VAL,   test_ids)

    # Sanity counts
    ev_count = len(list(out.rglob("events.h5")))
    args_count = len(list(out.rglob("v2e-script-args.txt")))
    print("events.h5 files in subset:", ev_count)
    print("v2e-script-args.txt files in subset:", args_count)

    print("Expected clips total:", len(train_small) + len(val_small) + len(test_small))
    # each clip should have 2 files
    print("Expected file total (approx):", 2 * (len(train_small) + len(val_small) + len(test_small)))

    if MAKE_ZIP:
        zip_subset_folder(out)

    print("Done. Subset folder:", out)

if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     ZIP_PATH = r"D:\Neuromorph data\20bn-jester-event.zip"
#     TRAIN_CSV = r"D:\Neuromorph data\20bn-jester-event\Train.csv"
#     VAL_CSV = r"D:\Neuromorph data\20bn-jester-event\Validation.csv"
#     OUT_DIR = r"D:\Neuromorph data\20bn-jester-event\Subset"

#     TRAIN_PER_CLASS = 300
#     VAL_PER_CLASS = 30
#     TEST_PER_CLASS = 30
#     SEED = 0
#     MAKE_ZIP = True

#     # --- call logic directly (same as main) ---
#     import pandas as pd
#     from pathlib import Path

#     out = Path(OUT_DIR)
#     out.mkdir(parents=True, exist_ok=True)

#     train_df = pd.read_csv(TRAIN_CSV)
#     val_df   = pd.read_csv(VAL_CSV)

#     print("Checking Train.csv labels...")
#     show_label_hints(train_df, TARGET_LABELS)
#     print("\nChecking Validation.csv labels...")
#     show_label_hints(val_df, TARGET_LABELS)

#     train_small, val_small, test_small = build_splits(
#         train_df, val_df,
#         seed=SEED,
#         train_per=TRAIN_PER_CLASS,
#         val_per=VAL_PER_CLASS,
#         test_per=TEST_PER_CLASS
#     )

#     train_small.to_csv(out / "Train_small.csv", index=False)
#     val_small.to_csv(out / "Validation_small.csv", index=False)
#     test_small.to_csv(out / "Test_small.csv", index=False)

#     train_ids = set(train_small["video_id"].astype(int).tolist())
#     val_ids   = set(val_small["video_id"].astype(int).tolist())
#     test_ids  = set(test_small["video_id"].astype(int).tolist())

#     print("\nExtracting train clips:", len(train_ids))
#     extract_needed_from_zip(ZIP_PATH, out, train_ids, r"Jester_Event\Train")

#     print("Extracting validation clips:", len(val_ids))
#     extract_needed_from_zip(ZIP_PATH, out, val_ids, r"Jester_Event\Validation")

#     print("Extracting test clips:", len(test_ids))
#     extract_needed_from_zip(ZIP_PATH, out, test_ids, r"Jester_Event\Validation")

#     if MAKE_ZIP:
#         subset_zip = out.parent / f"{out.name}.zip"
#         print("\nCreating zip:", subset_zip)
#         import zipfile
#         with zipfile.ZipFile(subset_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
#             for p in out.rglob("*"):
#                 if p.is_file():
#                     z.write(p, arcname=str(p.relative_to(out.parent)))
#         print("Done:", subset_zip)

#     print("\nSubset ready at:", out)
#     print("Train/Val/Test sizes:", len(train_small), len(val_small), len(test_small))
