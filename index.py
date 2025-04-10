
import controller.azure_blob_storage as azure_blob_storage
import controller.validate_img as validate_img
import os
from flask import Flask, request, render_template
from flask_cors import CORS

FACE_DIR = "images/face/"





app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # Check if the directory of Azure Bloc Storage exists
    azure_blob_storage.container_exists("bronze")
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    # Check the extension of the file
    image_format = validate_img.is_valid_image(file.filename)
    print(f"Image format: {image_format}")
    if not image_format:
        return 'Invalid image format', 400
    
    # Save the file to azure blob storage
    azure_blob_storage.upload_file_to_container("bronze", file.stream, file.filename)

    return f"Imagen {file.filename} recibida correctamente"

if __name__ == '__main__':
    app.run(debug=True)