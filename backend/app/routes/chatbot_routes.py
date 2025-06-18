from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime

from app.chatbot.chatbot import graph, init_components, start_qa, resume_qa
from app.models.chat_model import ChatMessage
from app.database import db
from app.utils.security import get_current_user

df, lexical_retrievers, semantic_retriever, query_llm, llm = init_components("./app/chatbot/scrapping_auto_df.csv", "./app/chatbot/halodoc_db", embedding_model="intfloat/multilingual-e5-large-instruct")#./app/chatbot/halodoc_db || intfloat/multilingual-e5-large-instruct

router = APIRouter()
chat_collection = db["chat_history"]

class ChatRequest(BaseModel):
    query: str
    threadid: Optional[str] = None

@router.post("/", summary="Tanya ke chatbot")
async def ask_chatbot(
    req: ChatRequest,
    user_id: str = Depends(get_current_user)
):
    if req.threadid:
        config = {
            "configurable": {
                "thread_id":req.threadid,
                "query_llm": query_llm,
                "llm": llm,
                "df": df,
                "lexical_retrievers": lexical_retrievers,
                "semantic_retriever": semantic_retriever
            }
        }

        result = resume_qa(question=req.query, graph=graph, config=config)
    else:
        config = {
            "configurable": {
                "thread_id":uuid4(),
                "query_llm": query_llm,
                "llm": llm,
                "df": df,
                "lexical_retrievers": lexical_retrievers,
                "semantic_retriever": semantic_retriever
            }
        }

        result = start_qa(question=req.query, graph=graph, config=config)

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

    if "__interrupt__" in result.keys(): 
        bot_msg = ChatMessage(
            id=uuid4(),
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=result["__interrupt__"][0].value,
            timestamp=now
        )
    else:
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

    if "__interrupt__" in result.keys():
        return {
            "answer": result["__interrupt__"][0].value,
            "thread_id": config["configurable"]["thread_id"]
        }
    else:
        return {
            "answer": result["answer"]
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
