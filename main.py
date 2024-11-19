from fastapi import FastAPI, HTTPException, UploadFile, File

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import numpy as np

import cv2

import base64

from PIL import Image

import io

import fitz

import logging

import os

from datetime import datetime

import sentry_sdk

from typing import Optional



# Configure logging

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

)

logger = logging.getLogger(__name__)



# Initialize Sentry (optional for error tracking)

if os.getenv("SENTRY_DSN"):

    sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"))



app = FastAPI(title="Document Scanner API")



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)



class ImageRequest(BaseModel):

    image: str



class ProcessingResponse(BaseModel):

    success: bool

    processed_image: Optional[str] = None

    error: Optional[str] = None



def decode_image(base64_string: str) -> np.ndarray:

    try:

        if 'base64,' in base64_string:

            base64_string = base64_string.split('base64,')[1]

        image_bytes = base64.b64decode(base64_string)

        nparr = np.frombuffer(image_bytes, np.uint8)

        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    except Exception as e:

        logger.error(f"Error decoding image: {str(e)}")

        raise HTTPException(status_code=400, detail="Invalid image format")



def detect_document_edges(image: np.ndarray) -> np.ndarray:

    try:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        

        if not contours:

            return None

            

        largest_contour = max(contours, key=cv2.contourArea)

        peri = cv2.arcLength(largest_contour, True)

        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

        

        if len(approx) == 4:

            return approx.reshape(4, 2)

        return None

    except Exception as e:

        logger.error(f"Error detecting edges: {str(e)}")

        return None



def order_points(pts: np.ndarray) -> np.ndarray:

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]

    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]

    rect[3] = pts[np.argmax(diff)]

    return rect



def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:

    try:

        rect = order_points(pts)

        (tl, tr, br, bl) = rect



        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        maxWidth = max(int(widthA), int(widthB))



        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        maxHeight = max(int(heightA), int(heightB))



        dst = np.array([

            [0, 0],

            [maxWidth - 1, 0],

            [maxWidth - 1, maxHeight - 1],

            [0, maxHeight - 1]

        ], dtype="float32")



        M = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        

        return warped

    except Exception as e:

        logger.error(f"Error in perspective transform: {str(e)}")

        return image



def enhance_document(image: np.ndarray) -> np.ndarray:

    try:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(

            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2

        )

        denoised = cv2.fastNlMeansDenoising(thresh)

        return denoised

    except Exception as e:

        logger.error(f"Error enhancing document: {str(e)}")

        return image



@app.get("/health")

async def health_check():

    return {

        "status": "healthy",

        "timestamp": str(datetime.now()),

        "version": "1.0.0"

    }



@app.post("/process-document")

async def process_document(request: ImageRequest):

    try:

        logger.info("Processing new document")

        

        # Decode image

        image = decode_image(request.image)

        

        # Detect edges

        corners = detect_document_edges(image)

        if corners is None:

            logger.warning("Could not detect document corners")

            return ProcessingResponse(

                success=False,

                error="Could not detect document corners"

            )

            

        # Transform perspective

        warped = four_point_transform(image, corners)

        

        # Enhance document

        enhanced = enhance_document(warped)

        

        # Convert to A4 size (2480 x 3508 pixels @ 300 DPI)

        a4_image = cv2.resize(enhanced, (2480, 3508))

        

        # Convert to PDF

        img_pil = Image.fromarray(a4_image)

        pdf_buffer = io.BytesIO()

        img_pil.save(pdf_buffer, format='PDF')

        

        # Convert to base64

        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode()

        

        logger.info("Document processed successfully")

        return ProcessingResponse(

            success=True,

            processed_image=pdf_base64

        )

        

    except Exception as e:

        logger.error(f"Error processing document: {str(e)}", exc_info=True)

        return ProcessingResponse(

            success=False,

            error=str(e)

        )



if __name__ == "__main__":

    import uvicorn

    port = int(os.getenv("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)