# Import necessary libraries
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from google.colab import files
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load the saved model
model = load_model('digit_recognition_model.h5')  # Make sure you upload your trained model here

# Function to preprocess the image (resize, normalize, reshape)
def preprocess_image(image):
    # Convert image to grayscale (if it's not already)
    grayscale_image = image.convert('L')
    
    # Resize the image to 28x28
    resized_image = grayscale_image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(resized_image) / 255.0
    
    # Reshape to match model's input shape (28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# Function to predict the digit
def predict_digit(uploaded_image):
    # Preprocess the image
    preprocessed_image = preprocess_image(uploaded_image)
    
    # Predict the digit
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)
    
    # Show the image and prediction
    plt.imshow(uploaded_image, cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.axis('off')
    plt.show()

    return predicted_digit

# Step 1: Upload an image file
uploaded = files.upload()

# Step 2: Read the uploaded image
for fn in uploaded.keys():
    image_path = fn
    image = Image.open(image_path)
    
    # Step 3: Predict the digit
    predicted_digit = predict_digit(image)
    print(f"Predicted Digit: {predicted_digit}")
