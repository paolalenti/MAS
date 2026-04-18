from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage
from pydantic import BaseModel
import uvicorn
import asyncio
from functools import partial

from core import app as langgraph_app
from ingest import run_ingestion
from database import Base, engine

api = FastAPI(title="AI Edu-Content Studio API")


class CourseRequest(BaseModel):
    topic: str


@api.post("/generate-course")
async def generate_course(request: CourseRequest):
    try:
        initial_state = {
            "topic": request.topic,
            "messages": [],
            "course_plan": [],
            "current_content": "",
            "quiz": {},
            "user_score": 0
        }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(langgraph_app.invoke, initial_state)
        )

        content = result.get("current_content", "")
        if not content:
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                    content = msg.content
                    break

        return {
            "status": "success",
            "topic": request.topic,
            "plan": result.get("course_plan", []),
            "content": content,
            "quiz": result.get("quiz", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    run_ingestion()
    Base.metadata.create_all(engine)
    print("Запуск API сервера на http://localhost:8000")
    uvicorn.run(api, host="0.0.0.0", port=8000)
