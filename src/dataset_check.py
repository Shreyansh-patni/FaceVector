from pathlib import Path
from PIL import Image

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_DIR = BASE_DIR / "data" / "att_faces" / "Training"
TESTING_DIR = BASE_DIR / "data" / "att_faces" / "Testing"


def inspect_dataset(folder, name):
    print(f"\n{'=' * 50}")
    print(f"{name} DATASET")
    print(f"{'=' * 50}")

    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        return

    image_files = list(folder.rglob("*.pgm"))

    print(f"Location: {folder}")
    print(f"Images found: {len(image_files)}")

    if not image_files:
        print("No .pgm images found.")
        return

    # Open first image
    first_image = image_files[0]

    with Image.open(first_image) as img:
        print(f"First image: {first_image.name}")
        print(f"Image format: {img.format}")
        print(f"Image size: {img.size}")
        print(f"Color mode: {img.mode}")

    print("\nFirst 5 images:")
    for image in image_files[:5]:
        print(f"  {image}")


inspect_dataset(TRAINING_DIR, "TRAINING")
inspect_dataset(TESTING_DIR, "TESTING")