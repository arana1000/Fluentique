from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import CORS_ORIGINS
from backend.routers.auth_router import router as auth_router
from backend.routers.conversation_router import router as conversation_router
from backend.routers.grammar_router import router as grammar_router
from backend.routers.quiz_router import router as quiz_router

# Import ensures Firebase is initialised when the backend starts.
from backend.utils import firebase_utils  # noqa: F401


# server.py
# ----------------------
# Main FastAPI application entry point for Fluentique.
#
# This file configures CORS, registers the application's routers, ensures
# Firebase is initialised at startup, and exposes a simple health endpoint.


app = FastAPI(
    title="Fluentique API",
    description="Backend API for the Fluentique language-learning application.",
)


# === CORS Configuration ===

def _get_allowed_origins(origins: str) -> list[str]:
    """
    Parse the configured comma-separated CORS origins.
    """
    if origins == "*":
        return ["*"]

    parsed_origins = [
        origin.strip()
        for origin in origins.split(",")
        if origin.strip()
    ]

    return parsed_origins or ["*"]


allowed_origins = _get_allowed_origins(
    CORS_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Routers ===

app.include_router(grammar_router)
app.include_router(auth_router)
app.include_router(quiz_router)
app.include_router(conversation_router)


# === Health Endpoint ===

@app.get("/")
def root() -> dict[str, str]:
    """Return a simple API health response."""
    return {
        "msg": "Fluentique backend running",
    }


# === Local Development Runner ===

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )