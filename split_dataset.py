import random
import shutil
from pathlib import Path

random.seed(42)

SRC_REAL = Path("training_real")
SRC_FAKE = Path("training_fake")
OUT_DIR = Path("dataset")

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_images(folder):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def make_dirs():
    for split in SPLITS:
        (OUT_DIR / split / "real").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "fake").mkdir(parents=True, exist_ok=True)

def split_and_copy(files, class_name):
    random.shuffle(files)
    n = len(files)

    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])
    n_test = n - n_train - n_val

    split_map = {
        "train": files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train+n_val:]
    }

    for split, split_files in split_map.items():
        for f in split_files:
            dst = OUT_DIR / split / class_name / f.name
            shutil.copy2(f, dst)

    print(f"{class_name}: total={n}, train={n_train}, val={n_val}, test={n_test}")

def main():
    if not SRC_REAL.exists() or not SRC_FAKE.exists():
        print(" Erreur: vérifie les dossiers training_real / training_fake")
        return

    real_files = get_images(SRC_REAL)
    fake_files = get_images(SRC_FAKE)

    print(f"Images real trouvées: {len(real_files)}")
    print(f"Images fake trouvées: {len(fake_files)}")

    if len(real_files) == 0 or len(fake_files) == 0:
        print(" Aucun fichier image trouvé.")
        return

    make_dirs()
    split_and_copy(real_files, "real")
    split_and_copy(fake_files, "fake")

    print("\n Split terminé dans le dossier 'dataset/'")

if __name__ == "__main__":
    main()