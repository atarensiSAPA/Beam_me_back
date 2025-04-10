import validate_img
import azure_blob_storage
import os
from flask import Flask, request, render_template
from flask_cors import CORS

FACE_DIR = "images/face/"

# Call the validate_image_format function to check if the image format is correct
# validate_img.validate_image_format(FACE_DIR)

# # Check if the directory of Azure Bloc Storage exists
# azure_blob_storage.container_exists("bronze")
# azure_blob_storage.upload_file_to_container("bronze", FACE_DIR)
# Dont do this, just for testing purposes, the file is in local: images/face/happy.jpg
# azure_blob_storage.download_file_from_container("bronze", "happy.jpg", "images/face/face.jpg")

# Get dataset from Github

app = Flask(__name__)
# Restricted only for the port 5500
CORS(app)


@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    # printar en la consola
    print("Recibido: ", file.filename)

if __name__ == '__main__':
    app.run(debug=True)