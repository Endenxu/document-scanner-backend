from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import base64
from PIL import Image
import io
import logging
import os

logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for more details
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ImageRequest(BaseModel):
    image: str

@app.post("/process-document")
async def process_document(request: ImageRequest):
    try:
        logger.debug("Starting document processing")
        
        # Decode base64 image
        try:
            if 'base64,' in request.image:
                request.image = request.image.split('base64,')[1]
            image_bytes = base64.b64decode(request.image)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            logger.debug(f"Image decoded successfully. Shape: {image.shape}")
        except Exception as e:
            logger.error(f"Image decoding error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug(f"Grayscale conversion complete. Shape: {gray.shape}")

        # Enhance image
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        logger.debug(f"Enhancement complete. Shape: {enhanced.shape}")

        # Convert to PIL Image
        pil_image = Image.fromarray(enhanced)
        
        # Create PDF
        pdf_buffer = io.BytesIO()
        pil_image.save(
            pdf_buffer, 
            format='PDF',
            resolution=300,
            quality=95
        )
        pdf_buffer.seek(0)
        logger.debug("PDF creation complete")

        # Convert to base64
        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode()
        logger.debug("Base64 conversion complete")

        return {
            "success": True,
            "processed_image": pdf_base64
        }

    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)