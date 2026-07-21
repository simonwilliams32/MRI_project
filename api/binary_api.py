from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
import cv2
import io
from PIL import Image
from mriproject.preproc_img import preprocess_image, get_model

app = FastAPI()

# # Allow all requests (optional, good for development purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    #nparr = np.fromstring(contents, np.uint8)
    #cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    cv2_img = Image.open(io.BytesIO(contents)) # should convert to image format

    ### Do cool stuff with your image.... For example face detection

    ### for testing just return the image
    img_preproc = preprocess_image(cv2_img)

    binary_model = get_model()

    pred = binary_model.predict(img_preproc)
    prob = float(pred[0][0])
    pred_class = "cancer" if prob > 0.5 else "healthy"
    output = {'class': pred_class, 'prob':prob}

    ### Encoding and responding with the image
    #im = cv2.imencode('.png', annotated_img)[1] # extension depends on which format is sent from Streamlit
    #return Response(content=im.tobytes(), media_type="image/png")
    return output
