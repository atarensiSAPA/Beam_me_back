from flask import Flask, request, render_template, jsonify
import controller.azure_blob_storage as azure_blob_storage
import controller.validate_img as validate_img
import controller.face_recognition as face_recognitionContrl
from flask_cors import CORS
import controller.hugging_face as hugging_face_model
import os
import shutil

def delete_folders(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # azure_blob_storage.container_exists("bronze", os.getenv("CONNECTION_STRING"))
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_format = validate_img.is_valid_image(file.filename)
    if not image_format:
        return jsonify({"error": "Invalid image format"}), 400
    
    # azure_blob_storage.upload_file_to_container("bronze", file.stream, file.filename, os.getenv("CONNECTION_STRING"))
    
    # Save the file locally for processing
    # create a directory if it doesn't exist
    if not os.path.exists('images/unkown'):
        os.makedirs('images/unkown')
    if not os.path.exists('images/detected_faces/'):
        os.makedirs('images/detected_faces/')
    file_path = f'images/unkown/{file.filename}'
    file.save(file_path)

    print(f"File saved to {file_path}")
    
    known_face_encodings, known_face_names = face_recognitionContrl.load_known_faces('images/known/')
    face_found = face_recognitionContrl.process_image(file_path, 'images/detected_faces/', known_face_encodings, known_face_names)
    
    if not face_found:
        print("No faces detected in the image.")
        delete_folders('images/unkown/')
        delete_folders('images/detected_faces/')
        return jsonify({"error": "No faces detected"}), 400
    
    # Train the model
    # hugging_face_model.train_model()
    
    # Classify the image using the trained model
    array_emotions = hugging_face_model.classify_image('images/detected_faces/')
    
    # Return the result as JSON
    return jsonify(array_emotions)
if __name__ == '__main__':
    app.run(debug=True)