from pydantic import BaseModel

from app.schemas.common_schema import DetectedFace


class DetectionResponse(BaseModel):
    image_width: int
    image_height: int
    faces: list[DetectedFace]
