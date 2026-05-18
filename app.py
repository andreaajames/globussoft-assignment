from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import json
import shutil
import os
import uuid
from deepface import DeepFace

app = FastAPI(title="Face Authentication API")

with open("model_config.json", "r") as f:
    config = json.load(f)

MODEL_NAME = config["model_name"]
DETECTOR = config["detector_backend"]
THRESHOLD = config["threshold"]

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


def save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[-1] or ".jpg"
    path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{ext}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


def get_embedding(img_path):
    result = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=False
    )
    return np.array(result[0]["embedding"])


def get_boxes(img_path):
    faces = DeepFace.extract_faces(
        img_path=img_path,
        detector_backend=DETECTOR,
        enforce_detection=False
    )
    boxes = []
    for face in faces:
        region = face.get("facial_area", {})
        if region:
            boxes.append({
                "x": region.get("x"),
                "y": region.get("y"),
                "w": region.get("w"),
                "h": region.get("h")
            })
    return boxes


def cosine_sim(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm != 0 else 0.0


@app.post("/verify")
async def verify(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    path1 = save_upload(image1)
    path2 = save_upload(image2)

    try:
        emb1 = get_embedding(path1)
        emb2 = get_embedding(path2)

        score = cosine_sim(emb1, emb2)
        label = "same person" if score >= THRESHOLD else "different person"

        boxes1 = get_boxes(path1)
        boxes2 = get_boxes(path2)

        return JSONResponse(content={
            "verification_result": label,
            "similarity_score": round(score, 4),
            "bounding_boxes": {
                "image_1": boxes1,
                "image_2": boxes2
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)


@app.get("/")
def root():
    return {"message": "Face Authentication API is running. POST two images to /verify"}
