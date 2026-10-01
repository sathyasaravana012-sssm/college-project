import os
import cv2
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Plant_disease_detection"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "real_world_dataset",
    "real_world_dataset",
    "Plantdoc-Dataset-master",
    "test"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "real_world_dataset",
    "real_world_dataset",
    "Plantdoc-Dataset-master",
    "plantdoc_preprocessed"
)

IMAGE_SIZE = (224, 224)

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PLANTDOC IMAGE PREPROCESSING")
print("=" * 70)

print("\nInput folder:")
print(INPUT_DIR)

print("\nOutput folder:")
print(OUTPUT_DIR)

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # 1. Convert BGR to RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # 2. Mild illumination / contrast correction
    #
    # CLAHE is applied only to the luminance channel.
    # This avoids aggressively changing leaf colors.
    # --------------------------------------------------------

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    lab = cv2.merge(
        (l_channel, a_channel, b_channel)
    )

    image = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    # --------------------------------------------------------
    # 3. Resize to model input size
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 4. Convert back to BGR for OpenCV saving
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    return image


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

total_images = 0
processed_images = 0
failed_images = 0

for class_name in sorted(os.listdir(INPUT_DIR)):

    class_input_dir = os.path.join(
        INPUT_DIR,
        class_name
    )

    if not os.path.isdir(class_input_dir):
        continue

    # Create corresponding class folder
    class_output_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    os.makedirs(
        class_output_dir,
        exist_ok=True
    )

    print(f"\nProcessing: {class_name}")

    for filename in sorted(
        os.listdir(class_input_dir)
    ):

        input_path = os.path.join(
            class_input_dir,
            filename
        )

        # Only process common image formats
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            continue

        total_images += 1

        try:

            # Read image
            image = cv2.imread(input_path)

            if image is None:
                raise ValueError(
                    "OpenCV could not read the image"
                )

            # Apply preprocessing
            processed = preprocess_image(image)

            # Save using same filename
            output_path = os.path.join(
                class_output_dir,
                filename
            )

            success = cv2.imwrite(
                output_path,
                processed
            )

            if not success:
                raise ValueError(
                    "Could not save processed image"
                )

            processed_images += 1

        except Exception as e:

            failed_images += 1

            print(
                f"  ERROR: {filename} -> {e}"
            )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print(f"Total images found     : {total_images}")
print(f"Successfully processed : {processed_images}")
print(f"Failed images          : {failed_images}")

print("\nProcessed dataset:")
print(OUTPUT_DIR)

print("=" * 70)