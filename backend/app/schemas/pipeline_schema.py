from pydantic import BaseModel

from app.schemas.anti_spoofing_schema import AntiSpoofingResult
from app.schemas.common_schema import DetectedFace
from app.schemas.emotion_schema import EmotionResult
from app.schemas.verification_schema import RecognitionResult


class FaceAnalysis(BaseModel):
    face: DetectedFace
    emotion: EmotionResult
    anti_spoofing: AntiSpoofingResult
    recognition: RecognitionResult


class FrameAnalysisResponse(BaseModel):
    image_width: int
    image_height: int
    faces: list[FaceAnalysis]
