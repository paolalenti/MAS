import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from functools import partial
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from core import app as langgraph_app
from database import Session as DBSession, Course, Module, Question, User

router = APIRouter()


def get_db():
    with DBSession() as session:
        yield session


def get_course_for_user(course_id: int, user_id: int, db: Session) -> Course:
    course = db.scalar(select(Course).where(Course.id == course_id))
    if not course:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")
    if course.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return course


def get_module_for_course(module_id: int, course: Course, db: Session) -> Module:
    module = db.scalar(
        select(Module).where(Module.id == module_id, Module.course_id == course.id)
    )
    if not module:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Module not found")
    return module


class UserCreate(BaseModel):
    username: str


class UserResponse(BaseModel):
    id: int
    username: str

    class Config:
        from_attributes = True


@router.get("/users/", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)):
    users = db.scalars(select(User)).all()
    return users


@router.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.id == user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    existing = db.scalar(select(User).where(User.username == body.username))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")

    user = User(username=body.username)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.id == user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Каскадное удаление: questions → modules → courses → user
    course_ids_result = db.scalars(select(Course.id).where(Course.user_id == user_id))
    course_ids = course_ids_result.all()

    if course_ids:
        module_ids = db.scalars(
            select(Module.id).where(Module.course_id.in_(course_ids))
        ).all()

        if module_ids:
            db.execute(delete(Question).where(Question.module_id.in_(module_ids)))
        db.execute(delete(Module).where(Module.course_id.in_(course_ids)))
        db.execute(delete(Course).where(Course.user_id == user_id))

    db.delete(user)
    db.commit()


@router.get("/users/{user_id}/courses")
def list_courses(user_id: int, db: Session = Depends(get_db)):
    courses = db.scalars(select(Course).where(Course.user_id == user_id)).all()
    return [
        {
            "id": c.id,
            "topic": c.topic,
            "modules_completed": c.modules_completed,
        }
        for c in courses
    ]


class CourseRequest(BaseModel):
    topic: str


@router.post("/users/{user_id}/courses")
async def generate_course(request: CourseRequest, user_id: str):
    try:
        initial_state = {
            "topic": request.topic,
            "messages": [],
            "course_plan": [],
            "current_module_index": 0,
            "current_content": "",
            "modules_content": {},
            "quiz": {},
            "all_quizzes": {},
            "user_score": 0,
            "tool_call_count": 0,
            "tool_messages_for_current_module": [],
        }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(langgraph_app.invoke, initial_state)
        )

        all_quizzes: dict = result.get("all_quizzes", {})
        modules_content: dict = result.get("modules_content", {})

        with DBSession.begin() as session:
            course = Course(user_id=user_id, topic=request.topic)
            session.add(course)
            session.flush()

            for module_topic in result.get("course_plan", []):
                module = Module(
                    course_id=course.id,
                    topic=module_topic,
                    content=modules_content.get(module_topic, ""),  # FIX: сохраняем контент модуля
                )
                session.add(module)
                session.flush()

                questions = all_quizzes.get(module_topic, [])

                if not isinstance(questions, list):
                    continue

                session.add_all([
                    Question(
                        module_id=module.id,
                        question_text=q["question"],
                        options=q["options"],
                        answer=q["answer"],
                    )
                    for q in questions
                    if {"question", "options", "answer"} <= q.keys()
                ])

        return {
            "status": "success",
            "topic": request.topic,
            "plan": result.get("course_plan", []),
            "modules_content": modules_content,
            "quizzes": all_quizzes,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/courses/{course_id}/modules")
def list_modules(user_id: int, course_id: int, db: Session = Depends(get_db)):
    course = get_course_for_user(course_id, user_id, db)
    modules = db.scalars(select(Module).where(Module.course_id == course.id)).all()
    return [
        {
            "id": m.id,
            "topic": m.topic,
            "completed": m.completed,
        }
        for m in modules
    ]


@router.get("/users/{user_id}/courses/{course_id}/modules/{module_id}")
def get_module(user_id: int, course_id: int, module_id: int, db: Session = Depends(get_db)):
    course = get_course_for_user(course_id, user_id, db)
    module = get_module_for_course(module_id, course, db)
    content = module["content"]

    return {
        "id": module.id,
        "course_id": module.course_id,
        "topic": module.topic,
        "content": content,
        "completed": module.completed,
    }


@router.get("/users/{user_id}/courses/{course_id}/modules/{module_id}/questions")
def list_questions(
    user_id: int, course_id: int, module_id: int, db: Session = Depends(get_db)
):
    course = get_course_for_user(course_id, user_id, db)
    module = get_module_for_course(module_id, course, db)
    questions = db.scalars(
        select(Question).where(Question.module_id == module.id)
    ).all()

    return [
        {
            "id": q.id,
            "question_text": q.question_text,
            "options": q.options,
        }
        for q in questions
    ]


class AnswerSubmission(BaseModel):
    answers: dict[int, int]


@router.post("/users/{user_id}/courses/{course_id}/modules/{module_id}/questions")
def submit_answers(
    user_id: int,
    course_id: int,
    module_id: int,
    body: AnswerSubmission,
    db: Session = Depends(get_db),
):
    course = get_course_for_user(course_id, user_id, db)
    module = get_module_for_course(module_id, course, db)

    questions = db.scalars(
        select(Question).where(Question.module_id == module.id)
    ).all()

    if not questions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No questions found")

    results = {}
    total = len(questions)
    correct_count = 0

    for q in questions:
        chosen = body.answers.get(q.id)
        if chosen is None:
            results[q.id] = {"correct": False, "reason": "Not answered"}
        else:
            is_correct = chosen == q.answer
            if is_correct:
                correct_count += 1
            results[q.id] = {
                "correct": is_correct,
                "correct_answer": q.answer,
            }

    all_correct = correct_count == total
    if all_correct and not module.completed:
        module.completed = True
        course.modules_completed += 1
        db.commit()

    return {
        "total": total,
        "correct": correct_count,
        "passed": all_correct,
        "results": results,
    }


@router.delete("/users/{user_id}/courses/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_course(user_id: int, course_id: int, db: Session = Depends(get_db)):
    course = get_course_for_user(course_id, user_id, db)

    module_ids = db.scalars(
        select(Module.id).where(Module.course_id == course.id)
    ).all()

    if module_ids:
        db.execute(delete(Question).where(Question.module_id.in_(module_ids)))
        db.execute(delete(Module).where(Module.course_id == course.id))

    db.delete(course)
    db.commit()
