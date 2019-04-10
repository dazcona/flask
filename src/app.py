#!/usr/bin/python

# Flask
from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from flask_bootstrap import Bootstrap
from werkzeug import secure_filename
# Keras
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
# Pandas
import pandas as pd
# Numpy
import numpy as np
# OS
import os
# Config
import config
# Sessions
from uuid import uuid4
# Time
import time


# APP
app = Flask(__name__)
app.secret_key = config.APP_KEY
Bootstrap(app)

# Static path
static_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"))

# Uploads path
uploads_path = os.path.join(static_path, "uploads")
app.config['UPLOAD_FOLDER'] = uploads_path

# Some parameteres
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
batch_size = 15


# MODEL
def load_model():
    # Clear session
    K.clear_session() 
    print('Loading JSON model..')
    with open(config.ML_MODEL, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    # load weights into new model
    print('Loading weights...')
    model.load_weights(config.MODEL_WEIGHTS)
    # model
    return model


def classify(model, filename_to_look):

    # Build dataframe
    test_filenames = [ os.path.join(app.config['UPLOAD_FOLDER'], filename_to_look) ]
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    nb_samples = test_df.shape[0]
    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        test_df, 
        app.config['UPLOAD_FOLDER'], 
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    # Predict!
    predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
    threshold = 0.5
    prob = predict[0]
    category = 1 if prob > threshold else 0
    name = 'dog 🐶' if category == 1 else 'cat 🐱'
    return name, prob


# LANDING
@app.route('/')
def index():
    return render_template('index.html')


# UPLOAD FILES
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' and 'capture' in request.files:
        # File
        f = request.files['capture']
        if f is None or f.filename.strip() == '':
            flash('No file selected!')
        else:
            # Create a unique "session ID" for this particular batch of uploads
            upload_key = str(uuid4())
            # Pattern for file names
            pattern = upload_key + '_' + time.strftime("%Y%m%d-%H%M%S")
            # New filename
            filename = secure_filename(pattern + "_" + f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Save
            f.save(path)
            print('"{}" saved as "{}"'.format(f.filename, filename))
            # Process
            return redirect(url_for('process', filename=filename))
    # GET
    return render_template('upload.html')


# PROCESS
@app.route("/process/<filename>")
def process(filename):
    # Load model
    model = load_model()
    # Predict!
    category, prob = classify(model, filename)
    path = os.path.join('uploads', filename)
    image = dict(path=path, name=filename, category=category, probability=prob)
    # VIEW
    return render_template('view.html', image=image)


if __name__ == '__main__':
    app.run(host='0.0.0.0')