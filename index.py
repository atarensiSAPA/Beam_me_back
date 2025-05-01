from flask import Flask, request, render_template, jsonify
import controller.azure_blob_storage as azure_blob_storage
import controller.validate_img as validate_img
import controller.face_recognition as face_recognitionContrl
from flask_cors import CORS
import controller.hugging_face as hugging_face_model
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # azure_blob_storage.container_exists("bronze", os.getenv("CONNECTION_STRING"))
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
    
    # azure_blob_storage.upload_file_to_container("bronze", file.stream, file.filename, os.getenv("CONNECTION_STRING"))
    
    # Save the file locally for processing
    # create a directory if it doesn't exist
    if not os.path.exists('images/unkown'):
        os.makedirs('images/unkown')
    file_path = f'images/unkown/{file.filename}'
    file.save(file_path)

    print(f"File saved to {file_path}")
    
    known_face_encodings, known_face_names = face_recognitionContrl.load_known_faces('images/known/')
    face_recognitionContrl.process_image(file_path, 'images/detected_faces/', known_face_encodings, known_face_names)
    
    # Train the model
    # hugging_face_model.train_model()
    
    # Classify the image using the trained model
    hugging_face_model.classify_image('images/detected_faces/')
    
    # Delete the file after processing
    os.remove(file_path)
    print(f"File deleted: {file_path}")
    
    return "Image processed successfully", 200
if __name__ == '__main__':
    app.run(debug=True)
