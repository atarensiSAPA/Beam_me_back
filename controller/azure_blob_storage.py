from dotenv import load_dotenv
import os
from azure.storage.blob import ContainerClient

load_dotenv(".env")

def container_exists(container_name):
    try:
        container_client = ContainerClient.from_connection_string(conn_str=os.getenv("CONNECTION_STRING"), container_name=container_name)
        if not container_client.exists():
            print(f"Container '{container_name}' does not exist. Creating it...")
            container_client.create_container()
            return False
        print(f"Container '{container_name}' exists.")
        return True
    except Exception as e:
        print(f"Error checking container existence: {e}")
        return False

def upload_file_to_container(container_name, folder_path):
    try:
        container_client = ContainerClient.from_connection_string(conn_str=os.getenv("CONNECTION_STRING"), container_name=container_name)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                blob_client = container_client.get_blob_client(blob=filename)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"File '{filename}' uploaded to container '{container_name}'.")
                
    except Exception as e:
        print(f"Error uploading file to container: {e}")
        
def download_file_from_container(container_name, blob_name, download_path):
    try:
        container_client = ContainerClient.from_connection_string(conn_str=os.getenv("CONNECTION_STRING"), container_name=container_name)
        blob_client = container_client.get_blob_client(blob=blob_name)

        with open(download_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        print(f"Blob '{blob_name}' downloaded to '{download_path}'.")
    except Exception as e:
        print(f"Error downloading file from container: {e}")