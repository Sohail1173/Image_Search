from flask import Flask, request, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image
from dev import Image_RAG
import io

app = Flask(__name__)
CORS(app)
limiter = Limiter(get_remote_address,app=app)

# Configure maximum file size (16 mb)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    # return '.' in filename and 
    return filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
# [1].lower()
    # in ALLOWED_EXTENSIONS

@app.route('/Search_Image', methods=['POST'])
@limiter.limit('100 per day')
def resize_image():
    try:
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400

        file = request.files['file']
        if file.filename == '':
            return {'error': 'No selected file'}, 400

        img = load_image(file)
        if img is None:
            return {'error': 'Invalid image file'}, 400

        width = int(request.form.get('width', 100))
        height = int(request.form.get('height', 100))

        if width <= 0 or height <= 0:
            return {'error': 'Invalid dimensions'}, 400

        resized_img = img.resize((width, height))

        img_io = io.BytesIO()
        resized_img.save(img_io, format=img.format or 'JPEG')
        img_io.seek(0)

        return send_file(
            img_io,
            mimetype=f'image/{img.format.lower() if img.format else "jpeg"}'
        )

    except Exception as e:
        return {'error': str(e)}, 500




