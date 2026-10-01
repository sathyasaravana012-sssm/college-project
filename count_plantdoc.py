import os

TEST_FOLDER = r"C:\Plant_disease_detection\real_world_dataset\real_world_dataset\Plantdoc-Dataset-master\test"

print("=" * 60)
print("PLANTDOC TEST DATASET")
print("=" * 60)

total = 0

for class_name in sorted(os.listdir(TEST_FOLDER)):
    class_path = os.path.join(TEST_FOLDER, class_name)

    if os.path.isdir(class_path):
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        print(f"{class_name:<40} {len(images)} images")
        total += len(images)

print("=" * 60)
print(f"TOTAL IMAGES: {total}")
print("=" * 60)