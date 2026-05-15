from pydantic import BaseModel


class NormalizedBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class DetectedFace(BaseModel):
    bbox: NormalizedBox
    detection_confidence: float
    crop_width: int
    crop_height: int
