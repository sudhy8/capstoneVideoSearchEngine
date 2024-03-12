from flask import Flask
import numpy as np
from sklearn.preprocessing import Binarizer
import json

import boto3

# Specify your AWS credentials
aws_access_key_id = 'AKIAXYKJWO4T5XRXMZ3A'
aws_secret_access_key = 'ESjsmIjVA77qPIWtsE7AtljYR5Ri6rXFfAvw0iLo'
aws_session_token = 'YOUR_SESSION_TOKEN'  # If applicable
# Create an S3 client with your credentials
s3 = boto3.client('s3', 
                  aws_access_key_id=aws_access_key_id, 
                  aws_secret_access_key=aws_secret_access_key, 
                  )

# List all buckets
response = s3.list_buckets()

# Print out the bucket names
print('Existing buckets:')
for bucket in response['Buckets']:
    print(f'  {bucket["Name"]}')


def upload_file(file_name, bucket_name, object_name=None):
    if object_name is None:
        object_name = file_name
    try:
        response = s3.upload_file(file_name, bucket_name, object_name)
        url = f'https://{bucket_name}.s3.amazonaws.com/{object_name}'
        # print(f'{file_name} uploaded successfully. URL: {url}')
        return url
    except Exception as e:
        print(f'Upload failed: {e}')

app = Flask(__name__)

def delete_file(bucket_name, object_name):
    try:
        response = s3.delete_object(Bucket=bucket_name, Key=object_name)
        return(f'{object_name} deleted successfully from {bucket_name}')
    except Exception as e:
        print(f'Deletion failed: {e}')

# Example usage


@app.route("/")
def upload():
    return upload_file('Moana.mp4', 'invideosearchbucket')

@app.route("/delete")
def delete():
    return delete_file('invideosearchbucket','Moana.mp4')