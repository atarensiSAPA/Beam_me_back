import os

def is_valid_image(image_path: str) -> bool:
    valid_extensions = ['.jpg', '.jpeg', '.png']
    _, ext = os.path.splitext(image_path)
    return ext.lower() in valid_extensions
