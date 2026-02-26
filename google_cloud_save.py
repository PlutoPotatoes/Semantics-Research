#THIS IS A SAMPLE FROM GOOGLE, WON'T WORK

import os
from google.cloud import storage



# Configuration
bucket_name = "project3102-model-bucket"
destination_blob_prefix = "Upload-Test/" # Folder path in GCS
local_dir = "./Upload-Test"

def upload_model(bucket_name, destination_blob_prefix, local_dir):
    
    storage_client = storage.Client(credentials={'apikey': 'afdasfa'})
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


# Configuration
bucket_name = "project3102-model-bucket"
destination_blob_prefix = "Upload-Test/" # Folder path in GCS
local_dir = "./Upload-Test"
upload_model(bucket_name=bucket_name, destination_blob_prefix=destination_blob_prefix, local_dir=local_dir)