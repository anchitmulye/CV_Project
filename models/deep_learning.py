import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model only once
model = load_model('saved_model/baseline_cnn_model.h5')

def predict_deep_learning(image_file):
    try:
        image = Image.open(image_file).convert('RGB').resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        class_label = 'potholes' if prediction > 0.5 else 'normal'
        confidence = prediction if prediction > 0.5 else 1 - prediction

        return class_label, confidence
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")
