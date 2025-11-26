"""
Database Seeder
Initialize database with sample data for all tables
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Dict, Any, List
import uuid
import datetime

from ..core.database import async_db
from ..schemas.db_models import (
    User, Course, Document, Content, Summary, RequestTracking, ApiLog,
    QuestionGeneration, AnswerKey, QuestionOption, GeneratedQuestion,
    Chatroom, Message, Response, UserContext, CourseKnowledgeBase
)
from ..core.logger import api_logger

router = APIRouter(prefix="/seed", tags=["database-seeder"])

class DatabaseSeeder:
    """Database seeder for creating initial data."""

    def __init__(self):
        self.seed_data = {
            "users": [
                {
                    "username": "admin",
                    "email": "admin@tutor.com",
                    "password_hash": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewzH1VvYUPLsnhzG",  # password: admin123
                    "role": "admin"
                },
                {
                    "username": "instructor1",
                    "email": "instructor1@university.edu",
                    "password_hash": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewzH1VvYUPLsnhzG",  # password: admin123
                    "role": "instructor"
                },
                {
                    "username": "student1",
                    "email": "student1@university.edu",
                    "password_hash": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewzH1VvYUPLsnhzG",  # password: admin123
                    "role": "student"
                },
                {
                    "username": "student2",
                    "email": "student2@university.edu",
                    "password_hash": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewzH1VvYUPLsnhzG",  # password: admin123
                    "role": "student"
                }
            ],
            "courses": [
                {
                    "title": "Introduction to Computer Science",
                    "description": "Fundamental concepts of programming and computer science"
                },
                {
                    "title": "Data Structures and Algorithms",
                    "description": "Advanced data structures and algorithmic thinking"
                },
                {
                    "title": "Machine Learning Fundamentals",
                    "description": "Introduction to machine learning concepts and applications"
                },
                {
                    "title": "Web Development",
                    "description": "Modern web development with HTML, CSS, JavaScript, and frameworks"
                }
            ]
        }

    async def seed_users(self) -> List[User]:
        """Seed users table."""
        from ..core.database import async_db

        users = []
        async with async_db.get_session() as session:
            for user_data in self.seed_data["users"]:
                # Check if user already exists
                result = await session.execute(
                    select(User).where(User.username == user_data["username"])
                )
                existing_user = result.scalar_one_or_none()

                if not existing_user:
                    user = User(**user_data)
                    session.add(user)
                    await session.flush()
                    await session.refresh(user)
                    users.append(user)
                    api_logger.info(f"✅ Created user: {user.username}")
                else:
                    users.append(existing_user)
                    api_logger.info(f"ℹ️ User already exists: {existing_user.username}")

            return users

    async def seed_courses(self, users: List[User]) -> List[Course]:
        """Seed courses table."""
        from ..core.database import async_db

        courses = []
        instructor = next((u for u in users if u.role == "instructor"), users[0])  # Use instructor or first user

        async with async_db.get_session() as session:
            for course_data in self.seed_data["courses"]:
                # Check if course already exists
                result = await session.execute(
                    select(Course).where(Course.title == course_data["title"])
                )
                existing_course = result.scalar_one_or_none()

                if not existing_course:
                    course_data["instructor_id"] = instructor.user_id
                    course = Course(**course_data)
                    session.add(course)
                    await session.flush()
                    await session.refresh(course)
                    courses.append(course)
                    api_logger.info(f"✅ Created course: {course.title}")
                else:
                    courses.append(existing_course)
                    api_logger.info(f"ℹ️ Course already exists: {existing_course.title}")

            return courses

    async def seed_documents(self, courses: List[Course]) -> List[Document]:
        """Seed documents table."""
        async with async_db.get_session() as session:
            documents = []
            sample_docs = [
                {
                    "title": "Python Basics.pdf",
                    "file_path": "/uploads/python_basics.pdf",
                    "file_type": "pdf",
                    "file_size": 1024000,
                    "md5_hash": "d41d8cd98f00b204e9800998ecf8427e",
                    "processing_status": "completed",
                    "has_embeddings": True,
                    "embedding_model": "text-embedding-3-small"
                },
                {
                    "title": "Algorithm Notes.pdf",
                    "file_path": "/uploads/algorithm_notes.pdf",
                    "file_type": "pdf",
                    "file_size": 2048000,
                    "md5_hash": "a1b2c3d4e5f6789012345678901234ab",
                    "processing_status": "completed",
                    "has_embeddings": True,
                    "embedding_model": "text-embedding-3-small"
                }
            ]

            for i, doc_data in enumerate(sample_docs):
                course = courses[i % len(courses)]  # Distribute documents across courses
                doc_data["course_id"] = course.course_id

                # Check if document already exists
                result = await session.execute(
                    select(Document).where(Document.md5_hash == doc_data["md5_hash"])
                )
                existing_doc = result.scalar_one_or_none()

                if not existing_doc:
                    document = Document(**doc_data)
                    session.add(document)
                    await session.flush()
                    await session.refresh(document)
                    documents.append(document)
                    api_logger.info(f"✅ Created document: {document.title}")
                else:
                    documents.append(existing_doc)
                    api_logger.info(f"ℹ️ Document already exists: {existing_doc.title}")

            return documents

    async def seed_contents(self, courses: List[Course]) -> List[Content]:
        """Seed contents table."""
        async with async_db.get_session() as session:
            contents = []
            sample_contents = [
                {
                    "title": "Introduction to Variables",
                    "content_text": "Variables are fundamental in programming. They store data values that can be referenced and manipulated in a program. In Python, variables are created when you assign a value to them.",
                    "content_type": "lecture",
                    "processing_status": "completed",
                    "source": "internal"
                },
                {
                    "title": "Algorithm Complexity",
                    "content_text": "Time complexity measures how the runtime of an algorithm grows as the input size increases. Common complexities include O(1), O(n), O(log n), O(n log n), O(n²), and O(2ⁿ).",
                    "content_type": "lecture",
                    "processing_status": "completed",
                    "source": "internal"
                }
            ]

            for i, content_data in enumerate(sample_contents):
                course = courses[i % len(courses)]

                # Check if content already exists
                result = await session.execute(
                    select(Content).where(Content.title == content_data["title"])
                )
                existing_content = result.scalar_one_or_none()

                if not existing_content:
                    content = Content(**content_data)
                    session.add(content)
                    await session.flush()
                    await session.refresh(content)
                    contents.append(content)
                    api_logger.info(f"✅ Created content: {content.title}")
                else:
                    contents.append(existing_content)
                    api_logger.info(f"ℹ️ Content already exists: {existing_content.title}")

            return contents

    async def seed_user_contexts(self, users: List[User], courses: List[Course]) -> List[UserContext]:
        """Seed user contexts table."""
        async with async_db.get_session() as session:
            contexts = []

            for user in users:
                if user.role in ["student"]:
                    for course in courses[:2]:  # Create contexts for first 2 courses
                        # Check if context already exists
                        result = await session.execute(
                            select(UserContext).where(
                                and_(
                                    UserContext.user_id == user.user_id,
                                    UserContext.course_id == course.course_id
                                )
                            )
                        )
                        existing_context = result.scalar_one_or_none()

                        if not existing_context:
                            context_data = {
                                "user_id": user.user_id,
                                "course_id": course.course_id,
                                "user_context": f"Student {user.username} learning {course.title}. Goals: understand core concepts, complete assignments, achieve good grades."
                            }
                            context = UserContext(**context_data)
                            session.add(context)
                            await session.flush()
                            await session.refresh(context)
                            contexts.append(context)
                            api_logger.info(f"✅ Created user context: {user.username} - {course.title}")
                        else:
                            contexts.append(existing_context)

            return contexts

    async def seed_chatrooms(self, users: List[User], courses: List[Course]) -> List[Chatroom]:
        """Seed chatrooms table."""
        async with async_db.get_session() as session:
            chatrooms = []

            for user in users:
                if user.role in ["student"]:
                    for course in courses[:2]:  # Create chatrooms for first 2 courses
                        # Check if chatroom already exists
                        result = await session.execute(
                            select(Chatroom).where(
                                and_(
                                    Chatroom.user_id == user.user_id,
                                    Chatroom.course_id == course.course_id
                                )
                            )
                        )
                        existing_chatroom = result.scalar_one_or_none()

                        if not existing_chatroom:
                            chatroom_data = {
                                "user_id": user.user_id,
                                "course_id": course.course_id,
                                "room_name": f"{user.username}'s {course.title} Chat",
                                "description": f"Chat room for {user.username} studying {course.title}",
                                "is_active": True,
                                "max_messages": 1000
                            }
                            chatroom = Chatroom(**chatroom_data)
                            session.add(chatroom)
                            await session.flush()
                            await session.refresh(chatroom)
                            chatrooms.append(chatroom)
                            api_logger.info(f"✅ Created chatroom: {chatroom.room_name}")
                        else:
                            chatrooms.append(existing_chatroom)

            return chatrooms

    async def seed_course_knowledge_bases(self, courses: List[Course], documents: List[Document]) -> List[CourseKnowledgeBase]:
        """Seed course knowledge bases table."""
        async with async_db.get_session() as session:
            knowledge_bases = []

            for course in courses:
                for doc in documents[:2]:  # Add first 2 documents to each course
                    # Check if knowledge base entry already exists
                    result = await session.execute(
                        select(CourseKnowledgeBase).where(
                            and_(
                                CourseKnowledgeBase.course_id == course.course_id,
                                CourseKnowledgeBase.material_id == doc.md5_hash
                            )
                        )
                    )
                    existing_kb = result.scalar_one_or_none()

                    if not existing_kb:
                        kb_data = {
                            "course_id": course.course_id,
                            "material_id": doc.md5_hash,
                            "material_type": "document",
                            "title": doc.title,
                            "file_name": doc.file_path.split("/")[-1],
                            "file_path": doc.file_path,
                            "file_size": doc.file_size,
                            "processed": True,
                            "embedding_model": doc.embedding_model,
                            "chunk_count": 50,  # Simulated chunk count
                            "access_count": 0
                        }
                        kb = CourseKnowledgeBase(**kb_data)
                        session.add(kb)
                        await session.flush()
                        await session.refresh(kb)
                        knowledge_bases.append(kb)
                        api_logger.info(f"✅ Added {doc.title} to {course.title} knowledge base")
                    else:
                        knowledge_bases.append(existing_kb)

            return knowledge_bases

    async def seed_all(self) -> Dict[str, Any]:
        """Seed all tables with sample data."""
        try:
            api_logger.info("🌱 Starting database seeding...")

            # Seed in order of dependencies
            users = await self.seed_users()
            courses = await self.seed_courses(users)
            documents = await self.seed_documents(courses)
            contents = await self.seed_contents(courses)
            user_contexts = await self.seed_user_contexts(users, courses)
            chatrooms = await self.seed_chatrooms(users, courses)
            knowledge_bases = await self.seed_course_knowledge_bases(courses, documents)

            summary = {
                "users": len(users),
                "courses": len(courses),
                "documents": len(documents),
                "contents": len(contents),
                "user_contexts": len(user_contexts),
                "chatrooms": len(chatrooms),
                "knowledge_base_entries": len(knowledge_bases)
            }

            api_logger.info(f"✅ Database seeding completed: {summary}")
            return {
                "success": True,
                "message": "Database seeded successfully",
                "summary": summary
            }

        except Exception as e:
            api_logger.error(f"❌ Database seeding failed: {e}")
            return {
                "success": False,
                "message": f"Database seeding failed: {str(e)}",
                "summary": {}
            }

# Initialize seeder
seeder = DatabaseSeeder()

@router.post("/all", status_code=201)
async def seed_database() -> Dict[str, Any]:
    """Seed all database tables with sample data."""
    return await seeder.seed_all()

@router.post("/users", status_code=201)
async def seed_users_only() -> Dict[str, Any]:
    """Seed only users table."""
    users = await seeder.seed_users()
    return {
        "success": True,
        "message": f"Seeded {len(users)} users",
        "count": len(users)
    }

@router.post("/courses", status_code=201)
async def seed_courses_only() -> Dict[str, Any]:
    """Seed only courses table."""
    users = await seeder.seed_users()  # Courses need users for instructor_id
    courses = await seeder.seed_courses(users)
    return {
        "success": True,
        "message": f"Seeded {len(courses)} courses",
        "count": len(courses)
    }

@router.get("/status")
async def get_seeding_status() -> Dict[str, Any]:
    """Get current database seeding status."""
    try:
        async with async_db.get_session() as session:
            # Count records in each table
            tables = {
                "users": len((await session.execute(select(User.user_id))).scalars().all()),
                "courses": len((await session.execute(select(Course.course_id))).scalars().all()),
                "documents": len((await session.execute(select(Document.document_id))).scalars().all()),
                "contents": len((await session.execute(select(Content.id))).scalars().all()),
                "user_contexts": len((await session.execute(select(UserContext.context_id))).scalars().all()),
                "chatrooms": len((await session.execute(select(Chatroom.chatroom_id))).scalars().all()),
                "knowledge_base_entries": len((await session.execute(select(CourseKnowledgeBase.id))).scalars().all())
            }

            return {
                "success": True,
                "message": "Database status retrieved",
                "tables": tables
            }

    except Exception as e:
        api_logger.error(f"Failed to get seeding status: {e}")
        return {
            "success": False,
            "message": f"Failed to get seeding status: {str(e)}",
            "tables": {}
        }

@router.delete("/all")
async def clear_all_data() -> Dict[str, str]:
    """Clear all data from database (use with caution!)."""
    try:
        api_logger.warning("⚠️ Clearing all database data...")

        async with async_db.get_session() as session:
            # Delete in order of dependencies (child tables first)
            tables_to_clear = [
                (CourseKnowledgeBase, "knowledge base entries"),
                (Response, "responses"),
                (Message, "messages"),
                (Chatroom, "chatrooms"),
                (UserContext, "user contexts"),
                (QuestionOption, "question options"),
                (AnswerKey, "answer keys"),
                (GeneratedQuestion, "generated questions"),
                (QuestionGeneration, "question generations"),
                (Document, "documents"),
                (Content, "contents"),
                (Summary, "summaries"),
                (RequestTracking, "request tracking"),
                (ApiLog, "API logs"),
                (Course, "courses"),
                (User, "users")
            ]

            for table, table_name in tables_to_clear:
                try:
                    await session.execute(table.__table__.delete())
                    api_logger.info(f"✅ Cleared {table_name}")
                except Exception as e:
                    api_logger.warning(f"⚠️ Could not clear {table_name}: {e}")

        api_logger.info("✅ All database data cleared")
        return {"message": "All database data cleared successfully"}

    except Exception as e:
        api_logger.error(f"❌ Failed to clear database: {e}")
        raise HTTPException(status_code=500, detail=str(e))