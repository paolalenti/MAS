from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from core import app as langgraph_app


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
            "user_score": 0
        }

        result = langgraph_app.invoke(initial_state)

        return {
            "status": "success",
            "topic": request.topic,
            "plan": result.get("course_plan", []),
            "content": result.get("current_content", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Запуск API сервера на http://localhost:8000")
    uvicorn.run(api, host="0.0.0.0", port=8000)
