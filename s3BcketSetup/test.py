from flask import Flask
import numpy as np
from sklearn.preprocessing import Binarizer
import json

import boto3



from transformers import PreTrainedTokenizerFast, CLIPProcessor, CLIPModel
import torch
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import weaviate
import os
import json

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

with weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL", "https://ccc-q2m6s9m3.weaviate.network"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY", "imlIdfNFZ8eC6rm4iCvSrTJyQeoBo6KjQtLT"))
) as client:
    print(client.is_ready())


    def img_embedder(images, names):
        client.connect()

        batch_size = 16
        image_arr = None

        for i in tqdm(range(0, len(images), batch_size)):
            # select batch of images
            batch = []
            for j in range(i, min(i + batch_size, len(images))):
                img = Image.open(images[j]).convert("RGB")
                batch.append(img)
            # process and resize
            batch = processor(
                text=None,
                images=batch,
                return_tensors='pt',
                padding=True
            )['pixel_values'].to(device)
            # get image embeddings
            batch_emb = model.get_image_features(pixel_values=batch)
            # convert to numpy array
            batch_emb = batch_emb.squeeze(0)
            batch_emb = batch_emb.cpu().detach().numpy().tolist()
            # add to larger array of all image embeddings
            if image_arr is None:
                image_arr = batch_emb
            else:
                image_arr += [batch_emb]

        videoSearch = client.collections.get("VideoSearch")
        for i in zip(image_arr, names):
            uuid = videoSearch.data.insert(
                {
                    "name": i[1]
                },
                vector=image_arr
            )
            print(uuid)
        print("Completed")

# img_embedder(images, names)
        


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
# response = s3.list_buckets()

# Print out the bucket names
# print('Existing buckets:')
# for bucket in response['Buckets']:
#     print(f'  {bucket["Name"]}')


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

# from transformers import CLIPProcessor, CLIPModel,PreTrainedTokenizerFast
from transformers import PreTrainedTokenizerFast, CLIPProcessor, CLIPModel,CLIPTokenizerFast
import torch
import weaviate.classes as wvc

device = "cuda" if torch.cuda.is_available() else \
("mps" if torch.backends.mps.is_available() else "cpu")

device = "cuda" if torch.cuda.is_available() else \
("mps" if torch.backends.mps.is_available() else "cpu")
model_id = "openai/clip-vit-base-patch32"

model = CLIPModel. from_pretrained(model_id) . to(device)
tokenizer = CLIPTokenizerFast. from_pretrained(model_id)
processor = CLIPProcessor. from_pretrained(model_id)

import weaviate
import os

with weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL", "https://ccc-q2m6s9m3.weaviate.network"),  # Replace with your WCS URL
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY", "imlIdfNFZ8eC6rm4iCvSrTJyQeoBo6KjQtLT"))  # Replace with your WCS key
) as client:  # Use this context manager to ensure the connection is closed
    print(client.is_ready())


def finder(keyword):
    client.connect()

    names = client.collections.get("videoSearch")

    inputs = tokenizer(keyword, return_tensors='pt')
    text_emb = model.get_text_features(**inputs)

    names = client.collections.get("videoSearch")

    response = names.query.near_vector(
            near_vector=list(text_emb[0]),
            limit=4,
            return_metadata=wvc.query.MetadataQuery(certainty=True)
        )
    outs=""
    for obj in response.objects:
        name = obj.properties['name']
        outs=outs+" "+name
        print(f"Name: {name}")

    return outs

@app.route("/")
def upload():
    return upload_file('Moana.mp4', 'invideosearchbucket')

@app.route("/delete")
def delete():
    return delete_file('invideosearchbucket','Moana.mp4')

@app.route("/convert")
def convert():
    img_embedder(['./bird.jpg'],['ImageBird2'])
    return "True"

@app.route("/search")
def search():
    return finder("hill")