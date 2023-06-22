from fastapi import FastAPI, UploadFile, File, Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Annotated
from datetime import date, datetime
import json
import cv2
import face_recognition
import os
import numpy as np
from matplotlib import pyplot as plt
from google.cloud import storage
import cv2


import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import numpy as np
import pandas as pd
import pickle
# import matplotlib.pyplot as plt
from code import download_blob
import json
from google.cloud import bigquery, storage
from google.oauth2 import service_account

from fastapi.responses import HTMLResponse
import pandas as pd
import os



bucket_name = 'emp_png'
client = storage.Client.from_service_account_json("cloudkarya-internship-415b6b4ef0ff.json")  
bucket = client.get_bucket(bucket_name)


def extract(request: Request):
    download_blob(bucket_name, source_file_name, dest_filename)

def list_images(bucket_name):
    blobs = client.list_blobs(bucket_name)
    images = []
    for blob in blobs:
        image_path = download_blob(bucket_name, blob.name, blob.name)
        images.append(image_path)
    return images
# key_path = "cloudkarya-internship-771681dff37f.json"
# bigquery_client = bigquery.Client.from_service_account_json(key_path)
# storage_client = storage.Client.from_service_account_json(key_path)

# project_id = "cloudkarya-internship"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get('/')  
def index(request : Request):
    context={"request" : request,
             "predictedtopic":"No Video"}
    return templates.TemplateResponse("index.html",context) 

@app.get("/main", response_class=HTMLResponse)
def lis( request : Request):
    images = list_images(bucket_name)  
    print(images)
    context = {"request": request, "images": images}
    return templates.TemplateResponse("index.html", context)    

@app.post("/upload_video", response_class=HTMLResponse)
async def upload_video(request : Request, video_file: UploadFile = File(...)):
    video_path = f"videos/{video_file.filename}"
    with open(video_path,"wb") as f:
        f.write(await video_file.read())
 


    a=extract_frames(video_path)   
    b=recognize_faces(a)
    #c=process_attendance_data(b)
    context = {
        "request": request, 
        "video_path": video_path,
        "b": b
    }
    return templates.TemplateResponse("index.html",context)


   



   


# def download_blob(bucket_name, source_file_name, dest_filename,storage_client):
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_file_name)
#     f = open(dest_filename,'wb')
#     blob.download_to_file(f)

#download_blob("emp_monitoring_videos_raw", "cloudkarya/model.pkl", "model.pkl",storage_client=client) 

with open('model.pkl', 'rb') as f:
    known_faces, known_names = pickle.load(f)
  


def extract_frames(video_path):
    print(f"Video = {video_path}")
    count = 0
    cap = cv2.VideoCapture(video_path)

    frame_counter = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        if count % 20 != 0:
            continue

        frames.append(frame)

    cap.release()
    return frames

from tempfile import TemporaryFile

# def process_file(event, context):
#     """Triggered by a change to a Cloud Storage bucket.
#     Args:
#          event (dict): Event payload.
#          context (google.cloud.functions.Context): Metadata for the event.
#     """

#     if event == None:
#         file_name='cloudkarya/20230616_0853.mp4'
#     else:
#         file_name = event['name']

#     print(f"Processing file: {file_name}.")


#     storage_client = storage.Client()

#     source_bucket = storage_client.bucket("emp_attendance_monitoring_raw")
#     source_blob = source_bucket.blob(file_name)
#     destination_bucket = client.bucket("emp_attendance_monitoring_processed")

#     download_video = file_name.split("/")[-1]
#     download_blob("emp_monitoring_videos_raw", file_name, download_video, storage_client=client)

#     # Extract frames from the video file.
#     frames = extract_frames(download_video)
#     frames_len = len(frames)
#     print(f"Number of frames = {frames_len}")
#     # Write the extracted frames to a new file in the destination bucket.
#     frame_counter = 1
#     for frame in frames:
#         destination_blob = destination_bucket.blob(f"frame_{frame_counter}.jpg")
#         with TemporaryFile() as gcs_image:
#             frame.tofile(gcs_image)
#             gcs_image.seek(0)
#             destination_blob.upload_from_file(gcs_image)
#         frame_counter += 1
#         print('Frames sent')
# process_file()



def recognize_faces(frames):
    attendance_dict = {}  # Dictionary to store attendance data

    for i, frame in enumerate(frames):
        # Get the original frame size
        width = frame.shape[1]
        height = frame.shape[0]

        # Calculate the cropping coordinates
        crop_x = (width - min(width, height)) // 2
        crop_y = (height - min(width, height)) // 2
        crop_width = min(width, height)
        crop_height = min(width, height)

        # Desired square frame size
        square_size = 500

        # Crop and resize frame
        cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
        resized_frame = cv2.resize(cropped_frame, (square_size, square_size))

        # Find faces in the frame
        face_locations = face_recognition.face_locations(resized_frame)
        face_encodings = face_recognition.face_encodings(resized_frame, face_locations)

        if len(face_locations) == 0:
            # Skip the frame if no faces are detected
            continue

        # Iterate over each detected face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare face encoding with the known faces
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            # Find the best match
            if len(matches) > 0:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                    # Update attendance dictionary with name and timestamp
                    attendance_dict[name] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

                # Draw a box around the face and label the name
                top, right, bottom, left = face_location
                cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(resized_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
 
        # Save the resulting frame as an image
        output_path = f'results/frame_{i}.jpg'
        cv2.imwrite(output_path, resized_frame) 
        html_table = "<table>\n"
        html_table += "<tr><th colspan='2'>Attendance</th></tr>\n"
        html_table += "<tr><th>Name</th><th>Date and Time</th></tr>\n"
        html_table += "</thead>\n" 
        for name, date in attendance_dict.items():
            html_table += f"<tr><td>{name}</td><td>{date}</td></tr>\n"
 
        html_table += "</table>"   
    return html_table   
   
  

# def process_attendance_data(attendance_dict):
#     # Convert the att endance dictionary to a DataFrame
#     df = pd.DataFrame.from_dict(attendance_dict, orient='index', columns=['Timestamp'])

#     # Split timestamp into separate date and time columns
#     df[['Date', 'Time']] = df['Timestamp'].str.split(' ', 1, expand=True)

#     # Remove the original timestamp column
#     df = df.drop("Timestamp", axis=1)

#     # Set the Entry/Exit column as 'Entry'
#     df['Entry/Exit'] = 'Entry'
#     df = df.sort_values("Time")
#     current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_file = f'Attendance_{current_datetime}.csv'
#     df.to_csv(output_file, index=True)
#     return output_file
