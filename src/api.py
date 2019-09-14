import time
from pathlib import Path
from PIL import Image

from flask import Flask, request, send_from_directory
from flask_cors import CORS

import torch
from werkzeug.utils import secure_filename

from src.constants import Constants
from src.utils.predict import load_network, predict_with_cam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('src/out/uploads')
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

CORS(app)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model_path = Path('src/out') / Constants.RUN_ID / \
    'checkpoints' / Constants.MODEL_NAME

network = load_network(model_path, device)


@app.route('/image/<path:image_name>', methods=['GET'])
def get_image(image_name):

    return send_from_directory(app.config['UPLOAD_FOLDER'].absolute(),
                               image_name,
                               as_attachment=True)


@app.route('/predict', methods=['POST'])
def get_result():

    if request.method == 'POST' and request.files['image']:

        run_id = str(int(time.time()))

        image = request.files['image']

        image_name = secure_filename(image.filename)

        original_save_path = f'{run_id}_{image_name}'
        img_save_path = app.config['UPLOAD_FOLDER'] / original_save_path

        image.save(str(img_save_path))

        prediction_result = predict_with_cam(
            network,
            img_save_path,
            final_conv_layer=Constants.FINAL_CONV_LAYER,
            fc_layer=Constants.FC_LAYER,
            device=device
        )

        predicted_save_path = f'{run_id}_predict_{image_name}'

        Image.fromarray(prediction_result['heatmap']).save(
            app.config['UPLOAD_FOLDER'] / predicted_save_path)

        if Constants.NEGATIVE_CLASS == prediction_result['label']:
            predicted_save_path = original_save_path

        result = {}
        result['original_image'] = str(original_save_path)
        result['predict_image'] = str(predicted_save_path)
        result['label'] = prediction_result['label']

        return result


if __name__ == '__main__':
    app.run(port=8000)
