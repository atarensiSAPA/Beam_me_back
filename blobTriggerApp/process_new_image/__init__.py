import logging
import azure.functions as func
from PIL import Image
import io

def main(blob: func.InputStream):
    logging.info(f"New blob detected: {blob.name}, Size: {blob.length} bytes")

    try:
        # Ejemplo: abrir imagen y leer metadatos
        image = Image.open(io.BytesIO(blob.read()))
        logging.info(f"Image format: {image.format}, Size: {image.size}")

        # Aquí puedes llamar tus propias funciones (por ejemplo face_recognitionContrl.detect_faces(...))

    except Exception as e:
        logging.error(f"Failed to process image: {e}")
