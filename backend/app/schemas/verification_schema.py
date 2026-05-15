from pydantic import BaseModel


class RecognitionResult(BaseModel):
    label: str
    confidence: float
    matched: bool


class RecognitionResponse(BaseModel):
    recognition: RecognitionResult


class RegisterResponse(BaseModel):
    person_id: str
    status: str
    message: str
