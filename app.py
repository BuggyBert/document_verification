from flask import Flask, request, jsonify
import joblib  # For loading your .pkl model file
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your trained model
model = joblib.load('random_forest_model.pkl')  # Make sure your model is in the same directory as app.py

@app.route('/verify', methods=['POST'])
def verify_document():
    # Get the image from the request
    img_file = request.files['image']
    img = Image.open(img_file.stream)
    
    # Preprocess the image (this part depends on your model's requirements)
    img = np.array(img)  # Convert image to numpy array (example)
    
    # Make prediction
    prediction = model.predict([img.flatten()])  # Flatten image and predict
    
    # Return the result (real or fake)
    return jsonify({'result': 'Real' if prediction[0] == 1 else 'Fake'})

if __name__ == '__main__':
    app.run(debug=True)
