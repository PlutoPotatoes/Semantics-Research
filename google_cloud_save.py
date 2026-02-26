#THIS IS A SAMPLE FROM GOOGLE, WON'T WORK

import os
from google.cloud import storage
from google.oauth2 import service_account
import json
import datasets


# Configuration
bucket_name = "project3102-model-bucket"
destination_blob_prefix = "Upload-Test/" # Folder path in GCS
local_dir = "./Upload-Test"

def upload_folder(credentials_path, bucket_name, destination_blob_prefix, local_dir):
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    # Upload each file in the local directory
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Determine the relative path to maintain folder structure in GCS
            relative_path = os.path.relpath(local_file_path, local_dir)
            remote_path = os.path.join(destination_blob_prefix, relative_path)
            
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{remote_path}")

    print("All model files uploaded to Google Cloud Storage.")

def upload_file(credentials_path, bucket_name, destination_folder, filepath):
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    #get folder or create if it doesn't exist
    blob = bucket.get_blob(destination_folder)
    if blob == None:
        blob = bucket.blob(destination_folder)
    blob.upload_from_filename(filepath)
    print(f"Uploaded {filepath} to gs://{bucket_name}")

def download_file(credentials_path, bucket_name, file_blob_name, download_path):
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.get_blob(blob_name=file_blob_name)
    if blob==None:
        raise FileNotFoundError(
            f"The blob {file_blob_name} does not exist."
        )
    
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    blob.download_to_filename(download_path)
    print(f"Downloaded gs://{bucket_name}/{file_blob_name} → {download_path}")

def get_data(credentials_path, bucket_name, data_blob_path):
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.get_blob(blob_name=data_blob_path)
    if blob==None:
        raise FileNotFoundError(
            f"The blob {data_blob_path} does not exist."
        )
    
    json_bytes = blob.download_as_bytes()
    json_data = json_bytes.decode('utf-8')
    data = json_data.__dict__
    ds = datasets.Dataset.from_dict(data)
    return ds



# Configuration
if __name__ == "__main__":
    bucket_name = "project3102-model-bucket"
    destination_blob_prefix = "Upload-Test/" # Folder path in GCS
    local_dir = "./Upload-Test"
    file = "Semantics-Research/Upload-Test/test.txt"
    upload_folder(credentials_path='./service_account.json', bucket_name=bucket_name, destination_blob_prefix=destination_blob_prefix, local_dir=local_dir)
    upload_file(credentials_path='./service_account.json', bucket_name=bucket_name, destination_folder=destination_blob_prefix, filepath=file)
    #download_file(credentials_path='./service_account.json', bucket_name=bucket_name, destination_folder=destination_blob_prefix, filepath=file)