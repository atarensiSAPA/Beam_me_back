from flask import Flask, request, render_template, jsonify
import controller.azure_blob_storage as azure_blob_storage
import controller.validate_img as validate_img
import controller.face_recognition as face_recognitionContrl
from flask_cors import CORS
import controller.hugging_face as hugging_face_model
import controller.api_jokes as api_jokes
import os
import shutil
import controller.changed_emotions as changed_emotions

def delete_folders(folder_path):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        except Exception as e:
            print(f"Failed to delete folder: {folder_path}")

def reset_folders(folder_path):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
            print(f"Reset folder: {folder_path}")
        except Exception as e:
            print(f"Failed to reset folder: {folder_path}")
    
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # azure_blob_storage.container_exists("bronze", os.getenv("CONNECTION_STRING"))
    return render_template('index.html')

from flask import send_from_directory
@app.route('/face_images/<path:filename>')
def face_images(filename):
    return send_from_directory('images/detected_faces', filename)

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
    if not os.path.exists('images/unknown'):
        os.makedirs('images/unknown')
    if not os.path.exists('images/detected_faces/'):
        os.makedirs('images/detected_faces/')
    else:
        # vaciar la carpeta
        reset_folders('images/detected_faces/')
    file_path = f'images/unknown/{file.filename}'
    file.save(file_path)

    print(f"File saved to {file_path}")
    
    known_face_encodings, known_face_names = face_recognitionContrl.load_known_faces('images/known/')
    face_found = face_recognitionContrl.process_image(file_path, 'images/detected_faces/', known_face_encodings, known_face_names)
    
    if not face_found:
        print("No faces detected in the image.")
        delete_folders('images/unknown/')
        delete_folders('images/detected_faces/')
        return jsonify({"error": "No faces detected"}), 400
    
    # Train the model
    # hugging_face_model.train_model()
    
    # Classify the image using the trained model
    array_emotions = hugging_face_model.classify_image('images/detected_faces/')
    
    delete_folders('images/unknown/')
    # delete_folders('images/detected_faces/')
    
    # Return the result as JSON
    return jsonify(array_emotions)

@app.route('/process_jokes', methods=['POST'])
def jokes():
    lengauge = request.get_data(as_text=True)
    print(lengauge)
    
    # Call the API to get a joke
    jokes_array = api_jokes.get_jokes(lengauge)
    
    if jokes_array:
        return jsonify({"joke": jokes_array}), 200
    else:
        return jsonify({"error": "Failed to fetch joke"}), 500
    
@app.route('/update_emotions', methods=['POST'])
def update_emotions():
    new_emotions = request.get_json()
    print(new_emotions)
    
    flag_new_emotions = changed_emotions.redirect_emotions(new_emotions)
    
    if not flag_new_emotions:
        return jsonify({"error": "Failed to update emotions"}), 500
    
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(debug=True)
    