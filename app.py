from flask import Flask, request, render_template
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load models
model_h5 = tf.keras.models.load_model('model.h5')
model_pkl = joblib.load('model.pkl')

# Helper function for image preprocessing
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))  # Adjust depending on model requirements
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route for welcome page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No file part'
    file = request.files['image']
    if file.filename == '':
        return 'No selected file'
    
    # Process the image
    image_bytes = file.read()
    image = preprocess_image(image_bytes)
    
    # Predict using the .h5 model
    prediction = model_h5.predict(image)
    result = "Benign" if prediction[0] < 0.5 else "Malignant"  # Example logic
    
    return render_template('result.html', result=result)

# Route for numerical input
@app.route('/enter_values', methods=['POST'])
def enter_values():
    # Get the numerical inputs from the form
    numerical_values = [float(value) for value in request.form.values()]
    numerical_values = np.array(numerical_values).reshape(1, -1)
    
    # Predict using the .pkl model
    prediction = model_pkl.predict(numerical_values)
    result = "Benign" if prediction[0] == 0 else "Malignant"  # Example logic
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
