from sqlalchemy import create_engine, Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base

# Настройки подключения (5433)
DATABASE_URL = "postgresql://edu_user:edu_password@localhost:5433/edu_memory"

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)

class CourseProgress(Base):
    __tablename__ = 'course_progress'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    topic = Column(String)
    current_module = Column(Integer, default=0)
    history = Column(JSON)


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Таблицы в PostgreSQL успешно созданы!")