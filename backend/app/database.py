from motor.motor_asyncio import AsyncIOMotorClient
from app.config import MONGO_URI, MONGO_DB

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]
