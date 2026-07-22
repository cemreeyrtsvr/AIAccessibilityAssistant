"""
FastAPI application entry point.
"""

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="VisionVoice AI",
    description="AI-powered accessibility assistant backend.",
    version="1.0.0",
)

app.include_router(router)