from flask import Flask, render_template, request
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model_path = 'models/papper_balls.keras'
model = load_model(model_path)

class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']  # Update class names as per your model

def predict(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Update target_size as per your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            img_path = os.path.join('static', file.filename)
            file.save(img_path)
            predicted_class_label, confidence = predict(model, img_path)
            return render_template('result.html', prediction=predicted_class_label, confidence=confidence, img_path=file.filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
