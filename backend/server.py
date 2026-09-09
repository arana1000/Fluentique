import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.auth_router import router as auth_router
from backend.routers.conversation_router import router as conversation_router
from backend.routers.grammar_router import router as grammar_router
from backend.routers.quiz_router import router as quiz_router

# Import ensures Firebase is initialised when the backend starts.
from backend.utils import firebase_utils  # noqa: F401


app = FastAPI(
    title="Fluentique API",
    description="Backend API for the Fluentique language-learning application.",
)


def _get_allowed_origins() -> list[str]:
    origins = os.getenv("CORS_ORIGINS", "*")

    if origins.strip() == "*":
        return ["*"]

    return [
        origin.strip()
        for origin in origins.split(",")
        if origin.strip()
    ]


allowed_origins = _get_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(grammar_router)
app.include_router(auth_router)
app.include_router(quiz_router)
app.include_router(conversation_router)


@app.get("/")
def root() -> dict[str, str]:
    """Return a simple API health response."""
    return {"msg": "Fluentique backend running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )