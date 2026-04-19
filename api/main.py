import uvicorn

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from database import Base, engine
from routes import router

app = FastAPI(
    title="AI Edu-Content Studio API",
    description="API for managing courses, modules and questions",
    version="1.0.0",
)
Instrumentator().instrument(app).expose(app)
app.include_router(router)


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Запуск API сервера на http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
