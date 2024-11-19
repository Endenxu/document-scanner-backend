from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import cv2
import base64
from PIL import Image
import io
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.get("/")
async def root():
    return HTMLResponse("""
        <html>
            <body>
                <h1>Document Scanner API</h1>
                <p>API is running. Use /process-document endpoint for document processing.</p>
                <p>Check /health for status.</p>
            </body>
        </html>
    """)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": str(datetime.now()),
        "version": "1.0.0"
    }

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

def enhance_document(image: np.ndarray) -> np.ndarray:
    try:
        height, width = image.shape[:2]
        logger.info(f"Original dimensions before enhancement: {width}x{height}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        processed_height, processed_width = denoised.shape[:2]
        logger.info(f"Dimensions after enhancement: {processed_width}x{processed_height}")
        
        return denoised
    except Exception as e:
        logger.error(f"Error enhancing document: {str(e)}")
        return image

@app.post("/process-document")
async def process_document(request: ImageRequest):
    try:
        logger.info("Processing new document")
        
        # Decode image
        image = decode_image(request.image)
        original_height, original_width = image.shape[:2]
        logger.info(f"Original image dimensions: {original_width}x{original_height}")
        
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        logger.info(f"Original aspect ratio: {aspect_ratio}")
        
        # Define target A4 dimensions (standard resolution)
        target_height = 3508  # A4 height at 300 DPI
        target_width = int(target_height * aspect_ratio)
        
        # Resize image maintaining aspect ratio
        resized = cv2.resize(image, (target_width, target_height))
        logger.info(f"Resized dimensions: {target_width}x{target_height}")
        
        # Enhance document
        enhanced = enhance_document(resized)
        
        # Convert to PDF with explicit DPI
        img_pil = Image.fromarray(enhanced)
        pdf_buffer = io.BytesIO()
        img_pil.save(pdf_buffer, format='PDF', resolution=300)
        
        # Convert to base64
        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode()
        
        logger.info("Document processed successfully")
        return {
            "success": True,
            "processed_image": pdf_base64
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)