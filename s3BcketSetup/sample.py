from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
import cv2
import random
from PIL import Image
import os


def split_video_into_scenes(video_path, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path) ## to get the video from video path
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
        print('    Scene %2d:  Start %s /  Frame %d, End %s / Frame %d' % (
            i+1,
            scene[0].get_timecode(), scene[0].get_frames(),
            scene[1].get_timecode(), scene[1].get_frames(),))
        # Get a random frame from the scene
        random_frame = random.randint(scene[0].get_frames(), scene[1].get_frames())
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        # Save the frame as an image file
        frame_file = f"frame_{i+1}.jpg"
        cv2.imwrite(frame_file, frame)
        scenes.append({
            'scene_number': i+1,
            'start_time': scene[0].get_timecode(),
            'end_time': scene[1].get_timecode(),
            'random_frame': random_frame,
            'frame_file': frame_file
        })
    print(type(split_video))
    print(split_video)
    cap.release()
    image_files = [scene['frame_file'] for scene in scenes]
    image_names = [f"Scene {scene['scene_number']}" for scene in scenes]
    return scenes
    # return scenes





print(split_video_into_scenes('Moana.mp4'))