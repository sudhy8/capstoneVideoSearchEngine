

import json
from flask import request
import time
import boto3
import os

from flask import Flask, Response
from flask_cors import CORS
from threading import Thread, Event


from dotenv import load_dotenv
load_dotenv()
aws_access_key_id = "AKIAXYKJWO4TVZYEHDFO"
aws_secret_access_key = "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"


from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import cv2
import random
from PIL import Image

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




def img_embedder(images, meta):
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

        videoSearch = client.collections.get("videoSearch")
        uuid = videoSearch.data.insert(
                meta,
                vector=image_arr
            )
        # for i in zip(image_arr, meta):
        #     uuid = videoSearch.data.insert(
        #         {
        #             "name": i[1]
        #         },
        #         vector=image_arr
        #     )
        print(uuid)
        print("Completed")
        
s3 = boto3.client('s3', 
                  aws_access_key_id=aws_access_key_id, 
                  aws_secret_access_key=aws_secret_access_key, 
                  )



def upload_file(file_name, bucket_name, object_name=None):
    print("Uploading....")

    if object_name is None:
        object_name = file_name
    try:
        response = s3.upload_file(file_name, bucket_name, object_name)
        url = f'https://{bucket_name}.s3.amazonaws.com/{object_name}'
        # print(f'{file_name} uploaded successfully. URL: {url}')
        return url
    except Exception as e:
        print(f'Upload failed: {e}')


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


with weaviate.connect_to_wcs(
    cluster_url= "https://invideosearchenginevectordb-w7tzhomw.weaviate.network",  
    auth_credentials=weaviate.auth.AuthApiKey("XxkC480IK053xgmZVdrjfQB5IHKWOtw0cVdA")
) as client: 
    print(client.is_ready())



def finder(keyword):
    client.connect()

    names = client.collections.get("videoSearch")

    inputs = tokenizer(keyword, return_tensors='pt')
    text_emb = model.get_text_features(**inputs)

    names = client.collections.get("videoSearch")

    response = names.query.near_vector(
            near_vector=list(text_emb[0]),
            limit=100,
            return_metadata=wvc.query.MetadataQuery(certainty=True)
        )
    # print(response)
    outs=[]
    for obj in response.objects:
        outs.append(obj.properties)

    return json.dumps(outs)





def split_video_into_scenes(video_path, video_name,threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path) ## to get the video from video path
    # return str(video_path)
    scene_manager = SceneManager() 
    scene_manager.add_detector(
        ContentDetector(threshold=35.0))  ## add/register a Scenedetector(here contentdetector) to run when scene detect is called.
    scene_manager.detect_scenes(video, show_progress=True,frame_skip =0) # frame_skip=0 by default
    scene_list = scene_manager.get_scene_list()
    split_video=split_video_ffmpeg(video_path, scene_list, show_progress=True)
    print("Number of Scenes: ",len(scene_list))
    #display the detected scene details
    scenes = []
    cap = cv2.VideoCapture(video_path)
    for i, scene in enumerate(scene_list):
        # print('    Scene %2d:  Start %s /  Frame %d, End %s / Frame %d' % (
        #     i+1,
        #     scene[0].get_timecode(), scene[0].get_frames(),
        #     scene[1].get_timecode(), scene[1].get_frames(),))
        # Get a random frame from the scene
        random_frame = random.randint(scene[0].get_frames(), scene[1].get_frames())
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        # Save the frame as an image file
        frame_file = f"{video_path}frame_{i+1}.jpg"
        cv2.imwrite(frame_file, frame)
        scenes.append({
            'scene_number': i+1,
            'start_time': scene[0].get_timecode(),
            'end_time': scene[1].get_timecode(),
            'random_frame': random_frame,
            'frame_file': frame_file
        })
        upload_file(frame_file,'invideosearchengineassets',frame_file)
        img_embedder([f"./{frame_file}"],{
            'scene_number': i+1,
            'start_time': scene[0].get_timecode(),
            'end_time': scene[1].get_timecode(),
            'random_frame': random_frame,
            'frame_file': frame_file,
            'video_name':video_name,
            'video_path':video_path
        })
    # print(type(split_video))
    # print(split_video)
    cap.release()
    # image_files = [scene['frame_file'] for scene in scenes]
    # image_names = [f"Scene {scene['scene_number']}" for scene in scenes]

   
   
    # return json.dumps(scenes)
    upload_file(video_path,'invideosearchengineassets')

    return "completed"



app = Flask(__name__)
CORS(app)

# Flag to track if the stream should be stopped
stop_stream = Event()

@app.route("/")
def upload():
    return "Hello"

@app.route("/search/<string:search_term>")
def search(search_term):
    return finder(search_term)



@app.route("/videoSplitter", methods=['POST'])
def split():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the uploaded file
    video_path = "___"+str(time.time())+"___"+file.filename
    video_name = file.filename

    file.save(video_path)
    
    # Process the uploaded file (e.g., save it, pass it to the splitting function)
    return split_video_into_scenes(video_path,video_name)


def stop_stream_after_10_seconds():
    stop_stream.wait(3)
    stop_stream.set()

@app.route('/stream')
def stream():
    if stop_stream.is_set():
        stop_stream.clear()

    def generate_data():
        while not stop_stream.is_set():
            data = {
                'time': time.time(),
                'value': 10 * (1 + 0.1 * (time.time() % 10))
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1)

    # Start the thread to stop the stream after 10 seconds
    stop_stream_thread = Thread(target=stop_stream_after_10_seconds)
    stop_stream_thread.start()

    return Response(generate_data(), mimetype='text/event-stream')


import boto3
from datetime import datetime
import json

@app.route("/randomVideos",methods=['GET'])
def findLatest():
# List the objects in the bucket, filtering for MP4 files
    response = s3.list_objects_v2(Bucket='invideosearchengineassets')
    # print("response:",response) 
    mp4_files = [obj for obj in response.get('Contents', []) if obj['Key'].endswith('.mp4')]

    # Sort the MP4 files by last modified date in descending order
    mp4_files.sort(key=lambda x: x['LastModified'], reverse=True)

    # Get the latest 20 MP4 files
    latest_mp4_files = []
    for obj in mp4_files[:20]:
        # Generate a presigned URL for the object
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': 'invideosearchengineassets',
                'Key': obj['Key']
            }
        )
        latest_mp4_files.append({
            'key': obj['Key'],
            'url': url,
            'last_modified': obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
        })

    # Convert the list of latest MP4 files to a JSON response
    json_response = json.dumps(latest_mp4_files, indent=2)
    return json_response

@app.route("/videoSplitter2", methods=['POST'])
def split2():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Save the uploaded file
    video_path = "___"+str(time.time())+"___"+file.filename
    video_name = file.filename

    file.save(video_path)
    
    # Process the uploaded file (e.g., save it, pass it to the splitting function)
    return Response(split_video_into_scenes2(video_path, video_name), mimetype='text/event-stream')

def split_video_into_scenes2(video_path, video_name,threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path) ## to get the video from video path
    yield "Starting upload_file(video_path,'invideosearchengineassets')"
    upload_file(video_path,'invideosearchengineassets')
    yield "Finished upload_file(video_path,'invideosearchengineassets')"
    
    scene_manager = SceneManager() 
    yield "Starting scene_manager.add_detector(ContentDetector(threshold=35.0))"
    scene_manager.add_detector(ContentDetector(threshold=35.0))  ## add/register a Scenedetector(here contentdetector) to run when scene detect is called.
    yield "Finished scene_manager.add_detector(ContentDetector(threshold=35.0))"
    
    scene_manager.detect_scenes(video, show_progress=True,frame_skip =0) # frame_skip=0 by default
    scene_list = scene_manager.get_scene_list()
    split_video=split_video_ffmpeg(video_path, scene_list, show_progress=True)
    yield f"Number of Scenes: {len(scene_list)}"
    
    #display the detected scene details
    scenes = []
    cap = cv2.VideoCapture(video_path)
    yield "Starting for i, scene in enumerate(scene_list):"
    for i, scene in enumerate(scene_list):
        # Get a random frame from the scene
        random_frame = random.randint(scene[0].get_frames(), scene[1].get_frames())
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        # Save the frame as an image file
        frame_file = f"{video_path}frame_{i+1}.jpg"
        cv2.imwrite(frame_file, frame)
        scenes.append({
            'scene_number': i+1,
            'start_time': scene[0].get_timecode(),
            'end_time': scene[1].get_timecode(),
            'random_frame': random_frame,
            'frame_file': frame_file
        })
        upload_file(frame_file,'invideosearchengineassets',frame_file)
        img_embedder([f"./{frame_file}"],{
            'scene_number': i+1,
            'start_time': scene[0].get_timecode(),
            'end_time': scene[1].get_timecode(),
            'random_frame': random_frame,
            'frame_file': frame_file,
            'video_name':video_name,
            'video_path':video_path
        })
    yield "Finished for i, scene in enumerate(scene_list):"
    
    cap.release()
    # image_files = [scene['frame_file'] for scene in scenes]
    # image_names = [f"Scene {scene['scene_number']}" for scene in scenes]
    yield split_video





