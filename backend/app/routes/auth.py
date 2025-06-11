from fastapi import APIRouter, HTTPException
from app.models.user_model import UserCreate, UserLogin
from app.utils.security import hash_password, verify_password, create_access_token
from app.database import db

router = APIRouter()

@router.post("/register")
async def register(user: UserCreate):
    existing = await db["users"].find_one({"email": user.email})
    if existing:
        raise HTTPException(400, "Email already registered")
    await db["users"].insert_one({
        "email": user.email,
        "hashed_password": hash_password(user.password)
    })
    return {"message": "User registered"}

@router.post("/login")
async def login(user: UserLogin):
    db_user = await db["users"].find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": str(db_user["_id"])})
    return {"access_token": token}
