"""
Application configuration.

All project settings are centralized here.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


# =========================
# PROJECT
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
IMAGE_DIR = ASSETS_DIR / "images"
SOUND_DIR = ASSETS_DIR / "sounds"

TEMP_DIR = PROJECT_ROOT / "temp"
LOG_DIR = PROJECT_ROOT / "logs"


# =========================
# API
# =========================

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

API_PREFIX = "/api/v1"


# =========================
# CAMERA
# =========================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

FPS = 30

# Bu ayar yalnızca yerel webcam kaynağı içindir; mobil istemci görüntülerine uygulanmaz.
WEBCAM_MIRROR_CORRECTION = os.getenv(
    "WEBCAM_MIRROR_CORRECTION",
    "True",
).lower() in {"1", "true", "yes"}


# =========================
# OCR
# =========================

TESSERACT_PATH = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

OCR_LANGUAGE = "eng"


# =========================
# LLM
# =========================

FOUNDRY_BASE_URL = os.getenv(
    "FOUNDRY_BASE_URL",
    "http://127.0.0.1:49675/v1"
)

FOUNDRY_API_KEY = os.getenv(
    "FOUNDRY_API_KEY",
    "none"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "phi-4-mini"
)

MAX_RESPONSE_TOKENS = 300

TEMPERATURE = 0.4


# =========================
# VISION
# =========================

CONFIDENCE_THRESHOLD = 0.50

DEVICE = "cuda"


# =========================
# SPEECH
# =========================

VOICE_RATE = 170

VOICE_VOLUME = 1.0


# =========================
# DEBUG
# =========================

DEBUG = os.getenv("DEBUG", "True") == "True"
