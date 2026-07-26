from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Dict, Optional, List

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    clinic_name: Optional[str] = Field(None, max_length=100)

    @field_validator('password')
    @classmethod
    def truncate_password(cls, v: str) -> str:
        if v:
            return v.encode('utf-8')[:72].decode('utf-8', errors='ignore')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

    @field_validator('password')
    @classmethod
    def truncate_password(cls, v: str) -> str:
        if v:
            return v.encode('utf-8')[:72].decode('utf-8', errors='ignore')
        return v

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

class UserProfile(BaseModel):
    id: str
    username: str
    email: str
    clinic_name: Optional[str] = None
    created_at: float

class UserProfileUpdate(BaseModel):
    password: Optional[str] = Field(None, min_length=6)
    clinic_name: Optional[str] = Field(None, max_length=100)

    @field_validator('password')
    @classmethod
    def truncate_password(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return v.encode('utf-8')[:72].decode('utf-8', errors='ignore')
        return v

class PredictionResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    filepath: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    gradcam_path: Optional[str] = None
    created_at: float

class ReportResponse(BaseModel):
    id: str
    prediction_id: str
    user_id: str
    pdf_path: str
    report_text: Dict
    created_at: float
