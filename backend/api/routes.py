"""
API Routes.
"""

import cv2
import numpy as np

from fastapi import (
    APIRouter,
    File,
    UploadFile,
)

from api.services import AIService

router = APIRouter()

service = AIService()


@router.get("/")
def root():

    return {
        "message": "VisionVoice AI Backend is running."
    }


@router.get("/health")
def health():

    return {
        "status": "ok"
    }


@router.post("/analyze")
async def analyze(
    image: UploadFile = File(...)
):

    contents = await image.read()

    np_array = np.frombuffer(
        contents,
        np.uint8,
    )

    frame = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR,
    )

    answer = service.analyze(frame)

    return {
        "response": answer
    }