import validate_img
import azure_blob_storage
import os

FACE_DIR = "images/face/"

# Call the validate_image_format function to check if the image format is correct
validate_img.validate_image_format(FACE_DIR)

# Check if the directory of Azure Bloc Storage exists
azure_blob_storage.container_exists("bronze")
azure_blob_storage.upload_file_to_container("bronze", FACE_DIR)
# Dont do this, just for testing purposes, the file is in local: images/face/happy.jpg
# azure_blob_storage.download_file_from_container("bronze", "happy.jpg", "images/face/face.jpg")

# Get dataset from Github
