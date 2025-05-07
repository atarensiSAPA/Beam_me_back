import os
import shutil
import requests
from urllib.parse import urlparse
from pathlib import Path
import uuid


def redirect_emotions(new_emotions_list):
    train_dir = 'facial_emotion_dataset/train_dir'

    for emotion_data in new_emotions_list:
        name = emotion_data.get("name")
        new_emotion = emotion_data.get("new_emotion")
        image_url = emotion_data.get("image_path")

        parsed_url = urlparse(image_url)
        filename = os.path.basename(parsed_url.path)
        new_filename = f"{uuid.uuid4().hex}_{os.path.basename(filename)}"

        local_image_path = os.path.join('images/detected_faces', filename)

        dest_dir = os.path.join(train_dir, new_emotion)
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, new_filename)
        try:
            shutil.copy(local_image_path, dest_path)
            print(f"Imagen de {filename} copiada a {dest_dir}")
        except Exception as e:
            print(f"Error copiando la imagen de {filename}: {e}")
            return False

    return True

def auto_train_model(model_name, model_path, connection_string):
    return False