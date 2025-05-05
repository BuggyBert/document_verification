from flask import Flask, request, jsonify
import joblib  # For loading your .pkl model file
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Load your trained model
model = joblib.load('random_forest_model.pkl')  # Make sure this file is in the same folder

@app.route('/verify', methods=['POST'])
def verify_document():
    try:
        # Get the image from the request
        img_file = request.files['image']
        img = Image.open(img_file.stream).convert('L')  # Convert to grayscale if needed
        
        # Resize or preprocess if needed (optional)
        img = img.resize((100, 100))  # Example: resize to 100x100
        img_array = np.array(img).flatten()  # Convert to 1D array

        # Make prediction
        prediction = model.predict([img_array])

        # Return the result
        result = 'Real' if prediction[0] == 1 else 'Fake'
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Required for Render hosting
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

