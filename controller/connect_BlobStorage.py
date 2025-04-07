from dotenv import load_dotenv
import os
import shutil
from azure.storage.filedatalake import DataLakeServiceClient
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from azure.storage.blob import ContainerClient
import pandas as pd
import json

load_dotenv(".env")

# Configuración de SQL
port = 1433

driver = 'ODBC Driver 17 for SQL Server'

connection_url = URL.create(
    "mssql+pyodbc",
    username=os.getenv("USERNAME"),
    password=os.getenv("PASSWORD"),
    host=os.getenv("HOST"),
    port=port,
    database=os.getenv("DATABASE"),
    query={"driver": driver}
)

service_client = DataLakeServiceClient.from_connection_string(conn_str=os.getenv("CONNECTION_STRING"))