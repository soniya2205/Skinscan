from flask import render_template, jsonify, Flask, redirect, url_for, request
import random
import os
import numpy as np
from keras.applications.mobilenet import MobileNet 
from tensorflow.keras.preprocessing import image
from PIL import Image
from PIL import ImageOps


from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import model_from_json
import keras
from keras import backend as K

app = Flask(__name__)

SKIN_CLASSES = {
  0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'Basal Cell Carcinoma',
  2: 'Benign Keratosis',
  3: 'Dermatofibroma',
  4: 'Melanoma',
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion'

}

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        path='static/data/'+f.filename
        f.save(path)
        # Open and preprocess the image
        img = image.load_img(path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        #model code
        j_file = open('modelnew.json', 'r')
        loaded_json_model = j_file.read()
        j_file.close()
        model = model_from_json(loaded_json_model)
        
        model.load_weights('model.h5')
        predictions = model.predict(img)
        pred_class = np.argmax(predictions)
        disease = SKIN_CLASSES[pred_class]
        accuracy = predictions[0][pred_class] * 100

        K.clear_session()
    return render_template('uploaded.html', title='Success', predictions=disease, acc=accuracy, img_file=f.filename)

if __name__ == "__main__":
    app.run(debug=True)