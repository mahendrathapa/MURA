import os
import time

from flask import Flask, request
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'src/out/uploads'


def create_folder(local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)


@app.route('/upload', methods=['POST'])
def upload_file():

    if request.method == 'POST' and request.files['image']:
        image = request.files['image']
        image_name = f'{int(time.time())}_{secure_filename(image.filename)}'
        create_folder(app.config['UPLOAD_FOLDER'])

        image.save(os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(image_name)))

        return "Success Fully uploaded"


if __name__ == '__main__':
    app.run(debug=True)
