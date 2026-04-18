import os
from sqlalchemy import create_engine, ForeignKey, ARRAY, String
from sqlalchemy.orm import sessionmaker, declarative_base, mapped_column, Mapped

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column()


class Course(Base):
    __tablename__ = 'courses'
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), index=True)
    topic: Mapped[str] = mapped_column()
    modules_completed: Mapped[int] = mapped_column(default=0)


class Module(Base):
    __tablename__ = 'modules'
    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey('courses.id'), index=True)
    topic: Mapped[str] = mapped_column()
    completed: Mapped[bool] = mapped_column(default=False)


class Question(Base):
    __tablename__ = 'questions'
    id: Mapped[int] = mapped_column(primary_key=True)
    module_id: Mapped[int] = mapped_column(ForeignKey('modules.id'), index=True)
    question_text: Mapped[str] = mapped_column()
    options: Mapped[list[str]] = mapped_column(ARRAY(String))
    answer: Mapped[int] = mapped_column()


user = os.getenv("POSTGRES_USER", "edu_user")
password = os.getenv("POSTGRES_PASSWORD", "edu_password")
db_name = os.getenv("POSTGRES_DB", "edu_memory")

db_host = os.getenv("POSTGRES_HOST", "localhost")
db_port = os.getenv("POSTGRES_PORT", "5433")

DATABASE_URL = f"postgresql://{user}:{password}@{db_host}:{db_port}/{db_name}"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(engine)


if __name__ == "__main__":
    from sqlalchemy import select

    Base.metadata.create_all(engine)

    # Write session - commits automatically
    with Session.begin() as session:
        user = User(username="test")
        session.add(user)

    # Readonly session - manually commit with session.commit()
    with Session() as session:
        user = session.scalar(
            select(User)
            .where(User.username == "test")
        )
        print(user)
