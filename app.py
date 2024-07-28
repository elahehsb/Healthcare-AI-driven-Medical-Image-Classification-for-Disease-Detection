from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('pneumonia_detection_model.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_path = 'uploads/' + file.filename
    file.save(img_path)
    
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    result = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
    
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
