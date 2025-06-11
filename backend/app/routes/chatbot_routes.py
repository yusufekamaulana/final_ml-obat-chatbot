from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List
from uuid import UUID, uuid4
from datetime import datetime

from app.chatbot.chatbot import graph, init_components
from app.models.chat_model import ChatMessage
from app.database import db
from app.utils.security import get_current_user

df, lexical_retrievers, semantic_retriever, query_llm, llm = init_components()

router = APIRouter()
chat_collection = db["chat_history"]

class ChatRequest(BaseModel):
    query: str

@router.post("/", summary="Tanya ke chatbot")
async def ask_chatbot(
    req: ChatRequest,
    user_id: str = Depends(get_current_user)
):
    state = {
        "df": df,
        "lexical_retrievers": lexical_retrievers,
        "semantic_retriever": semantic_retriever,
        "query_llm": query_llm,
        "llm": llm,
        "question": req.query
    }
    result = graph.invoke(state)

    now = datetime.utcnow()
    session_id = f"session-{user_id}"

    user_msg = ChatMessage(
        id=uuid4(),
        session_id=session_id,
        user_id=user_id,
        role="user",
        content=req.query,
        timestamp=now
    )

    bot_msg = ChatMessage(
        id=uuid4(),
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        content=result["answer"],
        timestamp=now
    )

    await chat_collection.insert_many([
        {**user_msg.dict(), "id": str(user_msg.id)},
        {**bot_msg.dict(), "id": str(bot_msg.id)}
    ])

    return {
        "answer": result["answer"],
        "context": [doc.page_content for doc in result["context"]]
    }

@router.get("/history", response_model=List[ChatMessage], summary="Ambil riwayat chat berdasarkan session_id")
async def get_chat_history(
    session_id: str = Query(...),
    user_id: str = Depends(get_current_user)
):
    cursor = chat_collection.find({"session_id": session_id, "user_id": user_id})
    results = []
    async for doc in cursor:
        doc["id"] = UUID(doc["id"]) if isinstance(doc["id"], str) else doc["id"]
        results.append(ChatMessage(**doc))
    return results

@router.post("/history", response_model=ChatMessage, summary="Simpan satu pesan manual ke riwayat")
async def save_chat_message(
    message: ChatMessage,
    user_id: str = Depends(get_current_user)
):
    await chat_collection.insert_one({
        **message.dict(),
        "id": str(message.id),
        "user_id": user_id
    })
    return message

@router.delete("/history/{id}", summary="Hapus pesan berdasarkan ID")
async def delete_message(
    id: UUID,
    user_id: str = Depends(get_current_user)
):
    result = await chat_collection.delete_one({"id": str(id), "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"detail": "Message deleted"}
