import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Selected classes
selected_classes = ['pizza', 'sushi', 'falafel', 'waffles', 'ice_cream']

# Nutrition info per food item
nutrition_info = {
    'pizza': {'calories': 285, 'protein': 12, 'carbs': 33, 'fat': 10},
    'sushi': {'calories': 200, 'protein': 5, 'carbs': 28, 'fat': 2},
    'falafel': {'calories': 400, 'protein': 13, 'carbs': 31, 'fat': 17},
    'waffles': {'calories': 350, 'protein': 6, 'carbs': 48, 'fat': 18},
    'ice_cream': {'calories': 270, 'protein': 3.5, 'carbs': 24, 'fat': 11}
}

# Burn durations in minutes (approx)
burn_methods = {
    'walking': {'pizza': 60, 'sushi': 40, 'falafel': 80, 'waffles': 70, 'ice_cream': 55},
    'running': {'pizza': 30, 'sushi': 20, 'falafel': 40, 'waffles': 35, 'ice_cream': 25},
    'cycling': {'pizza': 45, 'sushi': 30, 'falafel': 60, 'waffles': 50, 'ice_cream': 40},
    'yoga':    {'pizza': 75, 'sushi': 55, 'falafel': 100, 'waffles': 90, 'ice_cream': 70}
}

# Load the trained model from local file
model_path = "food_cnn_small.h5"

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found! Please put the model file in the project folder.")
    st.stop()

model = tf.keras.models.load_model(model_path)
IMG_SIZE = 224

# Preprocess image
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# App UI
st.set_page_config(page_title="SmartFit Nutrition Analyzer", layout="centered")
st.title("🍽️ SmartFit Nutrition Analyzer")
st.markdown("Upload a food image to get predictions, nutrition facts, and burn estimates.")

uploaded_file = st.file_uploader("📤 Upload Food Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='📸 Your Uploaded Image', use_container_width=True)

    img_array = preprocess_image(image)
    predictions = model.predict(img_array)[0]

    # Top 3 predictions
    top_indices = predictions.argsort()[-3:][::-1]
    st.subheader("🔍 Top 3 Predictions:")
    for i in top_indices:
        class_name = selected_classes[i]
        confidence = predictions[i] * 100
        st.markdown(f"- **{class_name.capitalize()}** — Confidence: `{confidence:.2f}%`")

    # Primary result
    top_class = selected_classes[top_indices[0]]
    nutrition = nutrition_info.get(top_class, None)

    st.markdown("---")
    st.subheader(f"🍔 Detected Food: **{top_class.capitalize()}**")

    if nutrition:
        st.write(f"🔥 **Calories:** {nutrition['calories']} kcal")
        st.write(f"💪 **Protein:** {nutrition['protein']} g")
        st.write(f"🍞 **Carbohydrates:** {nutrition['carbs']} g")
        st.write(f"🥑 **Fat:** {nutrition['fat']} g")
    else:
        st.write("Nutrition info not available.")

    st.markdown("### 🏃 Burn It Off")
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"🚶 Walking: {burn_methods['walking'][top_class]} min")
        st.success(f"🏃 Running: {burn_methods['running'][top_class]} min")

    with col2:
        st.warning(f"🚴 Cycling: {burn_methods['cycling'][top_class]} min")
        st.error(f"🧘 Yoga: {burn_methods['yoga'][top_class]} min")
