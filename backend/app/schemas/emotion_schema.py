from pydantic import BaseModel


class EmotionResult(BaseModel):
    label: str
    confidence: float


class EmotionResponse(BaseModel):
    emotion: EmotionResult
