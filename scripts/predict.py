import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import tensorflow as tf
import keras
from preprocess import load_data, preprocess_pixels



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results", "model", "final_emotion_model.keras")
TEST_CSV = os.path.join(BASE_DIR, "data", "test_with_emotions.csv")

model = load_model(MODEL_PATH)
print("===> Model Summary:")
print(model.summary())


data = load_data(TEST_CSV)

X_test = preprocess_pixels(data["pixels"])
y_test = data["emotion"].to_numpy()

predictions = model.predict(X_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

accuracy = float((y_pred == y_test).mean())

print(f"Accuracy on test set: {round(accuracy * 100)}%")
