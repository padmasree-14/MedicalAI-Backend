import os
import time
from datetime import datetime, timedelta
from typing import Optional
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from server.database.db import db_manager
from server.database.models import UserRegister, UserLogin, Token, UserProfile, UserProfileUpdate

# Hashing and Token configurations
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretjwtkeyforadvancedmedicalaiplatform123")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# Utility helpers
def hash_password(password: str) -> str:
    safe_pwd = password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
    return pwd_context.hash(safe_pwd)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    safe_pwd = plain_password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
    return pwd_context.verify(safe_pwd, hashed_password)

def create_access_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "exp": expire
    }
    encoded_jwt = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency to retrieve current logged in user from JWT bearer token"""
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = await db_manager.get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    return user

# REST Endpoints
@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user_in: UserRegister):
    # Check if user already exists
    existing_user = await db_manager.get_user_by_email(user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
        
    # Hash password and create record
    hashed_pwd = hash_password(user_in.password)
    user_dict = {
        "username": user_in.username,
        "email": user_in.email,
        "password_hash": hashed_pwd,
        "clinic_name": user_in.clinic_name or ""
    }
    
    created_user = await db_manager.create_user(user_dict)
    if not created_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user due to database constraint"
        )
        
    user_id = created_user.get("id") or created_user.get("_id", "")
    await db_manager.log_action(str(user_id), "auth_register", f"User {user_in.username} registered.")
    return {"message": "User registered successfully", "user_id": str(user_id)}

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    user = await db_manager.get_user_by_email(credentials.email)
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
        
    # Generate token
    token = create_access_token(user["id"])
    await db_manager.log_action(user["id"], "auth_login", f"User logged in successfully.")
    return {"access_token": token, "token_type": "bearer"}

@router.get("/profile", response_model=UserProfile)
async def get_profile(current_user: dict = Depends(get_current_user)):
    return UserProfile(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        clinic_name=current_user.get("clinic_name"),
        created_at=current_user["created_at"]
    )

@router.post("/profile", response_model=UserProfile)
async def update_profile(profile_update: UserProfileUpdate, current_user: dict = Depends(get_current_user)):
    update_data = {}
    if profile_update.clinic_name is not None:
        update_data["clinic_name"] = profile_update.clinic_name
    if profile_update.password is not None:
        update_data["password_hash"] = hash_password(profile_update.password)
        
    if not update_data:
        raise HTTPException(status_code=400, detail="No values provided for update")
        
    success = await db_manager.update_user(current_user["id"], update_data)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update profile")
        
    # Get refreshed profile
    updated_user = await db_manager.get_user_by_id(current_user["id"])
    await db_manager.log_action(current_user["id"], "auth_update_profile", "User profile details updated.")
    
    return UserProfile(
        id=updated_user["id"],
        username=updated_user["username"],
        email=updated_user["email"],
        clinic_name=updated_user.get("clinic_name"),
        created_at=updated_user["created_at"]
    )
