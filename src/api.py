import os
import time
from pathlib import Path
from PIL import Image

from flask import Flask, request
import torch
from werkzeug.utils import secure_filename

from src.constants import Constants
from src.utils.predict import load_network, predict_with_cam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'src/out/uploads'

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model_path = Path('src/out') / Constants.RUN_ID / \
    'checkpoints' / Constants.MODEL_NAME

network = load_network(model_path, device)


def make_dir(upload_dir):

    upload_path = (Path(upload_dir)) / str(int(time.time()))
    upload_path.mkdir(parents=True, exist_ok=True)

    return upload_path


@app.route('/predict', methods=['POST'])
def get_result():

    if request.method == 'POST' and request.files['image']:

        predictions_path = make_dir(app.config['UPLOAD_FOLDER'])

        image = request.files['image']

        image_name = secure_filename(image.name)
        img_save_path = predictions_path / secure_filename(image_name)
        image.save(str(img_save_path))

        prediction_result = predict_with_cam(
            network,
            img_save_path,
            final_conv_layer=Constants.FINAL_CONV_LAYER,
            fc_layer=Constants.FC_LAYER,
            device=device
        )

        original_save_path = predictions_path / "original_image.png"
        predicted_save_path = predictions_path / "predicted_image.png"

        Image.fromarray(prediction_result['image']).convert(
            'L').save(original_save_path)

        Image.fromarray(prediction_result['heatmap']).save(predicted_save_path)

        result = {}
        result['original_image'] = os.path.abspath(original_save_path)
        result['heatmap_image'] = os.path.abspath(predicted_save_path)
        result['label'] = prediction_result['label']

        return result


if __name__ == '__main__':
    app.run(debug=True)
