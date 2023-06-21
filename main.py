from fastapi import FastAPI, Request, File
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from google.oauth2 import service_account
from code import download_blob
from conv import conve

bucket_name = 'emp_attendance_monitoring_processed'
client = storage.Client.from_service_account_json("cloudkarya-internship-415b6b4ef0ff.json")
bucket = client.get_bucket(bucket_name)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def dynamic_file(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def extract(request: Request):
    download_blob(bucket_name, source_file_name, dest_filename)

def list_images(bucket_name):
    blobs = client.list_blobs(bucket_name)
    images = []
    for blob in blobs:
        image_path = download_blob(bucket_name, blob.name, blob.name)
        conve(image_path)
        images.append(image_path)
    return images

@app.get("/main")
def index( request : Request):
    images = list_images(bucket_name)
    print(images)
    context = {"request": request, "images": images}
    return templates.TemplateResponse("index.html", context)