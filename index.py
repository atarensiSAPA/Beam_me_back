from flask import Flask, request, render_template, jsonify
import controller.azure_blob_storage as azure_blob_storage
import controller.validate_img as validate_img
import controller.face_recognition as face_recognitionContrl
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    azure_blob_storage.container_exists("bronze")
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    image_format = validate_img.is_valid_image(file.filename)
    if not image_format:
        return 'Invalid image format', 400
    
    azure_blob_storage.upload_file_to_container("bronze", file.stream, file.filename)

    return 'Image uploaded and processed successfully', 200
if __name__ == '__main__':
    app.run(debug=True)
