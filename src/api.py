from typing import Annotated

from fastapi import FastAPI, File

from utils import load_config, load_facenet

from src.workflow.prediction import predict_image




app = FastAPI()

@app.post("/image/", status_code=201)
async def load_image(img_fl: Annotated[bytes, File()]):

    config = load_config()

    model = load_facenet(config["model_path"])

    state = predict_image()

    results = {
        "status": "success"
    }

    return results