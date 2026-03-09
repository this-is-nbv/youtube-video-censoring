from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.inference import run_inference

app = FastAPI(title="MFusTSVD Video Moderation API")


# -----------------------------
# CORS (needed for extension)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow browser extension
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request model
# -----------------------------
class VideoRequest(BaseModel):
    video_id: str


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "Backend running"}


@app.get("/ping")
def ping():
    print("🔥 PING RECEIVED")
    return {"msg": "pong"}


# -----------------------------
# Main analysis endpoint
# -----------------------------
@app.post("/analyze")
def analyze(req: VideoRequest):

    try:
        result = run_inference(req.video_id)

        # Ensure expected fields exist
        response = {
            "label": result.get("label"),
            "profanity_windows": result.get("profanity_windows", []),
            "profane_words": result.get("profane_words", [])
        }

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )