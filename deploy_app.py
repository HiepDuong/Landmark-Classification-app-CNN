from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import torchvision.transforms as T
import torch

app = FastAPI()

# Load the trained model
learn_inf = torch.jit.load("checkpoints/transfer_exported.pt")

@app.post("/predict/")
async def predict_landmark(file: UploadFile):
    try:
        # Load the uploaded image
        image_stream = io.BytesIO(await file.read())
        img = Image.open(image_stream)

        # Transform to tensor
        timg = T.ToTensor()(img).unsqueeze_(0)

        # Get predictions
        softmax = learn_inf(timg).data.cpu().numpy().squeeze()
        idxs = np.argsort(softmax)[::-1]
        
        results = []
        for i in range(5):
            p = softmax[idxs[i]]
            landmark_name = learn_inf.class_names[idxs[i]]
            results.append({"landmark": landmark_name, "probability": float(p)})

        return JSONResponse(content={"predictions": results}, status_code=200)
    
    except Exception as e:
        return HTTPException(detail=str(e), status_code=400)