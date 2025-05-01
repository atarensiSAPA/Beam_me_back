from dotenv import load_dotenv
import os
from azure.storage.blob import ContainerClient
import uuid

load_dotenv(".env")

def container_exists(container_name, connection_string):
    try:
        container_client = ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name)
        if not container_client.exists():
            print(f"Container '{container_name}' does not exist. Creating it...")
            container_client.create_container()
            return False
        print(f"Container '{container_name}' exists.")
        return True
    except Exception as e:
        print(f"Error checking container existence: {e}")
        return False

def upload_file_to_container(container_name, file_stream, filename, connection_string):
    try:
        container_client = ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name)
        default_name = f"{uuid.uuid4().hex}_{os.path.basename(filename)}"
        blob_client = container_client.get_blob_client(default_name)
        blob_client.upload_blob(file_stream, overwrite=True)
        print(f"File '{filename}' uploaded to container '{container_name}'.")
                
    except Exception as e:
        print(f"Error uploading file to container: {e}")