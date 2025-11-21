import streamlit as st
import numpy as np
from PIL import Image
import requests
import io

st.set_page_config(page_title="Teachable Machine Webcam App", layout="centered")

st.title("🤖 Teachable Machine Image Model (Streamlit Version)")
st.write("Show your gesture or object to the camera and let the model predict!")

# ==========================
# 1. LOAD MODEL FROM TM LINK
# ==========================

MODEL_URL = "https://teachablemachine.withgoogle.com/models/P3-HT5rh8/"

@st.cache_resource
def load_model():
    model_path = MODEL_URL + "model.json"

    # Download model.json
    model_json = requests.get(model_path).json()

    # Convert model.json → h5 weights (TM exports URLs internally)
    model = tf.keras.models.model_from_json(str(model_json))

    # Load weights from *.bin files
    for layer in model.layers:
        if hasattr(layer, "set_weights"):
            weights_url = MODEL_URL + layer.name + ".bin"
            try:
                weights = requests.get(weights_url)
                weights_bytes = io.BytesIO(weights.content)
                layer_weights = np.load(weights_bytes, allow_pickle=True)
                layer.set_weights(layer_weights)
            except:
                pass

    return model

with st.spinner("Loading Teachable Machine model..."):
    model = load_model()
st.success("Model loaded successfully!")

# ==========================
# 2. LOAD LABELS
# ==========================

def load_labels():
    labels_url = MODEL_URL + "metadata.json"
    meta = requests.get(labels_url).json()
    return [item["className"] for item in meta["labels"]]

labels = load_labels()
st.write("🟢 **Model Classes:**")
st.write(labels)

# ==========================
# 3. CAMERA INPUT
# ==========================

img_file = st.camera_input("📸 Capture Image")

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Captured Image", width=300)

    # ==========================
    # 4. PREPROCESS IMAGE
    # ==========================
    img_resized = image.resize((224, 224))
    img_array = np.asarray(img_resized).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)

    # ==========================
    # 5. PREDICT
    # ==========================
    with st.spinner("Predicting..."):
        predictions = model.predict(img_array)[0]

    # ==========================
    # 6. DISPLAY RESULTS
    # ==========================
    st.subheader("🔮 Prediction Results")
    for label, prob in zip(labels, predictions):
        st.write(f"**{label}** — {prob:.2%}")

    st.success(f"🎉 Best Match: **{labels[np.argmax(predictions)]}**")
