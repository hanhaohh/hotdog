import os
import glob
from flask import Flask
from flask import jsonify
from flask import request, render_template
import numpy as np
from hotdog import process_image
from keras.models import load_model


model = load_model('../model.h5')
hotdog_image = process_image("static/img/cat.png", (128, 128))
hotdog_image1 = np.expand_dims(hotdog_image, axis=0)
p = model.predict(hotdog_image1)
tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=tmp_dir)
app.config['UPLOAD_FOLDER'] = 'static/img'

valid_mimetypes = ['image/jpeg', 'image/png']


@app.route('/')
def index():
    recent_files = sorted(
        glob.glob("%s/*" % app.config['UPLOAD_FOLDER']),
        key=os.path.getctime, reverse=True
    )
    slice_index = 2 if len(recent_files) > 1 else len(recent_files)
    recents = recent_files[:slice_index]
    return render_template('index.html', recents=recents)

"""
Endpoint for hot dog prediction
"""
@app.route('/is-hot-dog', methods=['POST'])
def is_hot_dog():
    if request.method == 'POST':
        if not 'file' in request.files:
            return jsonify({'error': 'no file'}), 400
        # Image info
        img_file = request.files.get('file')
        img_name = img_file.filename
        mimetype = img_file.content_type
        if not mimetype in valid_mimetypes:
            return jsonify({'error': 'bad-type'})
        image_name = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        img_file.save(image_name)
        hotdog_image = process_image(image_name, (128, 128))
        hotdog_image1 = np.expand_dims(hotdog_image, axis=0)
        # with graph:
        hot_dog_conf = model.predict(hotdog_image1)[0]

        # Delete image when done with analysis
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
        is_hot_dog = 'false' if hot_dog_conf < 0.5 else 'true'
        return_packet = {
            'is_hot_dog': is_hot_dog,
            'confidence': float(hot_dog_conf)
        }
        return jsonify(return_packet)
