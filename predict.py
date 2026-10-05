import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================================
# 1. Settings
# ==========================================

MODEL_PATH = r".\model\plant_disease_model.keras"
IMAGE_PATH = r".\test_images\test5.jpg"

IMAGE_SIZE = (224, 224)

# Exact class order used during model training
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ==========================================
# 2. Check image exists
# ==========================================

if not os.path.exists(IMAGE_PATH):
    print("ERROR: Image not found!")
    print("Put an image inside the test_images folder.")
    print("Current path:", IMAGE_PATH)
    exit()


# ==========================================
# 3. Load model
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Model output classes:", model.output_shape[-1])


# ==========================================
# 4. Load uploaded/test image
# ==========================================

print("\nLoading image...")

image = Image.open(IMAGE_PATH).convert("RGB")

print("Original image size:", image.size)


# ==========================================
# 5. Preprocess image
# ==========================================

# Central crop then resize to match training
width, height = image.size
target = min(width, height)
left = (width - target) // 2
top = (height - target) // 2
right = left + target
bottom = top + target
image = image.crop((left, top, right, bottom))
image = image.resize(IMAGE_SIZE)

image_array = np.array(image, dtype=np.float32)

# Convert pixel values from 0-255 to 0-1
image_array = image_array / 255.0

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

print("Processed image shape:", image_array.shape)


# ==========================================
# 6. Make prediction
# ==========================================

print("\nMaking prediction...")

predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])

confidence = predictions[0][predicted_index] * 100

predicted_class = CLASS_NAMES[predicted_index]


# ==========================================
# 7. Display result
# ==========================================

print("\n===================================")
print("          PREDICTION")
print("===================================")

print("Class index :", predicted_index)
print("Prediction  :", predicted_class)
print("Confidence  :", f"{confidence:.2f}%")

print("===================================")


# ==========================================
# 8. Show Top 5 predictions
# ==========================================

print("\nTop 5 Predictions:")
print("-----------------------------------")

top_indices = np.argsort(predictions[0])[-5:][::-1]

for index in top_indices:
    probability = predictions[0][index] * 100
    print(f"{CLASS_NAMES[index]} : {probability:.2f}%")