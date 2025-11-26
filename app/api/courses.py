"""
Course Management API
CRUD operations for course management with proper database integration
"""
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import time

from ..core.database import async_db, AsyncCRUD
from ..schemas.db_models import Course, User, Document, Chatroom, UserContext
from ..core.logger import api_logger

router = APIRouter(prefix="/courses", tags=["courses"])

# Pydantic models for request/response
class CourseCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Course title")
    description: Optional[str] = Field(None, max_length=2000, description="Course description")
    instructor_id: str = Field(..., description="Instructor user ID")

class CourseUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255, description="Updated course title")
    description: Optional[str] = Field(None, max_length=2000, description="Updated course description")
    instructor_id: Optional[str] = Field(None, description="Updated instructor user ID")

class CourseResponse(BaseModel):
    course_id: str
    title: str
    description: Optional[str]
    instructor_id: str
    created_at: str
    updated_at: str
    instructor_name: Optional[str] = None
    instructor_email: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None

class CourseListResponse(BaseModel):
    courses: List[CourseResponse]
    total_count: int
    page: int
    limit: int

class CourseStatsResponse(BaseModel):
    course_id: str
    documents_count: int
    chatrooms_count: int
    user_contexts_count: int
    total_file_size: int
    processed_documents: int
    pending_documents: int

# CRUD operations using AsyncCRUD base class
class CourseCRUD(AsyncCRUD):
    def __init__(self):
        super().__init__(Course)

    async def get_courses_by_instructor(
        self,
        instructor_id: str,
        page: int = 1,
        limit: int = 20,
        include_stats: bool = False
    ) -> tuple[List[Dict[str, Any]], int]:
        """Get paginated courses for a specific instructor."""
        async with async_db.get_session() as session:
            # Build base query
            base_conditions = [Course.instructor_id == instructor_id]

            # Get total count
            count_result = await session.execute(
                select(func.count(Course.course_id)).where(and_(*base_conditions))
            )
            total_count = count_result.scalar()

            # Get paginated results with instructor info
            result = await session.execute(
                select(Course, User.username, User.email)
                .join(User, Course.instructor_id == User.user_id)
                .where(and_(*base_conditions))
                .order_by(desc(Course.updated_at))
                .limit(limit)
                .offset((page - 1) * limit)
            )

            courses_with_instructors = []
            for row in result.fetchall():
                course, username, email = row
                courses_with_instructors.append({
                    'course': course,
                    'instructor_name': username,
                    'instructor_email': email
                })

            return courses_with_instructors, total_count

    async def get_course_with_details(self, course_id: str) -> Optional[Dict[str, Any]]:
        """Get course with instructor and statistics details."""
        async with async_db.get_session() as session:
            result = await session.execute(
                select(Course, User.username, User.email)
                .join(User, Course.instructor_id == User.user_id)
                .where(Course.course_id == course_id)
            )

            row = result.first()
            if not row:
                return None

            course, username, email = row

            # Get course statistics
            # Get total documents
            docs_count_result = await session.execute(
                select(func.count(Document.document_id))
                .where(Document.course_id == course_id)
            )
            documents_count = docs_count_result.scalar() or 0

            # Get total file size
            file_size_result = await session.execute(
                select(func.coalesce(func.sum(Document.file_size), 0))
                .where(Document.course_id == course_id)
            )
            total_file_size = file_size_result.scalar() or 0

            # Get processed documents count
            processed_result = await session.execute(
                select(func.count(Document.document_id))
                .where(and_(
                    Document.course_id == course_id,
                    Document.processing_status == "completed"
                ))
            )
            processed_documents = processed_result.scalar() or 0

            # Get pending documents count
            pending_result = await session.execute(
                select(func.count(Document.document_id))
                .where(and_(
                    Document.course_id == course_id,
                    Document.processing_status == "pending"
                ))
            )
            pending_documents = pending_result.scalar() or 0

            # Get chatroom and user context counts
            chatroom_result = await session.execute(
                select(func.count(Chatroom.chatroom_id))
                .where(Chatroom.course_id == course_id)
            )
            chatrooms_count = chatroom_result.scalar()

            context_result = await session.execute(
                select(func.count(UserContext.context_id))
                .where(UserContext.course_id == course_id)
            )
            contexts_count = context_result.scalar()

            # Combine all stats
            stats = {
                "documents_count": documents_count,
                "total_file_size": total_file_size,
                "processed_documents": processed_documents,
                "pending_documents": pending_documents,
                "chatrooms_count": chatrooms_count,
                "user_contexts_count": contexts_count
            }

            return {
                "course": course,
                "instructor_info": {
                    "username": username,
                    "email": email
                },
                "stats": stats
            }

    async def search_courses(
        self,
        query: str,
        page: int = 1,
        limit: int = 20
    ) -> tuple[List[Course], int]:
        """Search courses by title or description."""
        async with async_db.get_session() as session:
            # Build search query
            search_pattern = f"%{query}%"

            # Get total count
            count_result = await session.execute(
                select(func.count(Course.course_id))
                .where(
                    or_(
                        Course.title.ilike(search_pattern),
                        Course.description.ilike(search_pattern)
                    )
                )
            )
            total_count = count_result.scalar()

            # Get paginated results with instructor info
            result = await session.execute(
                select(Course, User.username, User.email)
                .join(User, Course.instructor_id == User.user_id)
                .where(
                    or_(
                        Course.title.ilike(search_pattern),
                        Course.description.ilike(search_pattern)
                    )
                )
                .order_by(desc(Course.updated_at))
                .limit(limit)
                .offset((page - 1) * limit)
            )

            courses = []
            for row in result.fetchall():
                course, username, email = row
                courses.append(course)

            return courses, total_count

    async def get_popular_courses(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get popular courses based on chatroom usage."""
        async with async_db.get_session() as session:
            result = await session.execute(
                select(
                    Course.course_id,
                    Course.title,
                    Course.description,
                    Course.created_at,
                    User.username,
                    func.count(Chatroom.chatroom_id).label("chatrooms_count"),
                    func.count(UserContext.context_id).label("contexts_count")
                )
                .join(User, Course.instructor_id == User.user_id)
                .join(Chatroom, Course.course_id == Chatroom.course_id, isouter=True)
                .join(UserContext, Course.course_id == UserContext.course_id, isouter=True)
                .group_by(Course.course_id, Course.title, Course.description, Course.created_at, User.username)
                .order_by(
                    desc(func.count(Chatroom.chatroom_id)),
                    desc(func.count(UserContext.context_id))
                )
                .limit(limit)
            )

            popular_courses = []
            for row in result.fetchall():
                (course_id, title, description, created_at, username,
                 chatrooms_count, contexts_count) = row

                popular_courses.append({
                    "course_id": course_id,
                    "title": title,
                    "description": description,
                    "created_at": created_at,
                    "instructor_name": username,
                    "chatrooms_count": chatrooms_count,
                    "contexts_count": contexts_count,
                    "popularity_score": chatrooms_count + contexts_count
                })

            return popular_courses

# Initialize CRUD
course_crud = CourseCRUD()

@router.post("/", response_model=CourseResponse, status_code=201)
async def create_course(course_data: CourseCreate) -> CourseResponse:
    """Create a new course."""
    try:
        api_logger.info(f"Creating course: {course_data.title}")

        # Check if instructor exists
        async with async_db.get_session() as session:
            user_result = await session.execute(
                select(User).where(User.user_id == course_data.instructor_id)
            )
            user = user_result.scalar_one_or_none()

            if not user:
                raise HTTPException(status_code=404, detail=f"Instructor {course_data.instructor_id} not found")

            if user.role not in ["instructor", "admin"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"User {course_data.instructor_id} is not an instructor"
                )

        # Check if course with same title already exists for this instructor
        async with async_db.get_session() as session:
            existing_result = await session.execute(
                select(Course).where(
                    and_(
                        Course.title == course_data.title,
                        Course.instructor_id == course_data.instructor_id
                    )
                )
            )
            existing_course = existing_result.scalar_one_or_none()

            if existing_course:
                raise HTTPException(
                    status_code=409,
                    detail=f"Course '{course_data.title}' already exists for this instructor"
                )

        # Create course
        course_dict = course_data.model_dump()
        course = await course_crud.create(course_dict)

        api_logger.info(f"✅ Created course {course.course_id}")

        # Get instructor details for response
        async with async_db.get_session() as session:
            instructor_result = await session.execute(
                select(User.username, User.email).where(User.user_id == course_data.instructor_id)
            )
            instructor_data = instructor_result.first()

        return CourseResponse(
            course_id=course.course_id,
            title=course.title,
            description=course.description,
            instructor_id=course.instructor_id,
            instructor_name=instructor_data.username if instructor_data else None,
            instructor_email=instructor_data.email if instructor_data else None,
            created_at=course.created_at.isoformat(),
            updated_at=course.updated_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to create course: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{course_id}", response_model=Dict[str, Any])
async def get_course(course_id: str) -> Dict[str, Any]:
    """Get course by ID with instructor and statistics details."""
    try:
        course_details = await course_crud.get_course_with_details(course_id)

        if not course_details:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        course = course_details["course"]

        return {
            "course_id": course.course_id,
            "title": course.title,
            "description": course.description,
            "instructor_id": course.instructor_id,
            "instructor_info": course_details["instructor_info"],
            "stats": course_details["stats"],
            "created_at": course.created_at.isoformat(),
            "updated_at": course.updated_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get course {course_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{course_id}", response_model=CourseResponse)
async def update_course(course_id: str, course_data: CourseUpdate) -> CourseResponse:
    """Update course by ID."""
    try:
        # Check if course exists
        async with async_db.get_session() as session:
            result = await session.execute(
                select(Course).where(Course.course_id == course_id)
            )
            course = result.scalar_one_or_none()

            if not course:
                raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        # Validate instructor_id if provided
        if course_data.instructor_id:
            async with async_db.get_session() as session:
                user_result = await session.execute(
                    select(User).where(User.user_id == course_data.instructor_id)
                )
                user = user_result.scalar_one_or_none()

                if not user:
                    raise HTTPException(status_code=404, detail=f"Instructor {course_data.instructor_id} not found")

                if user.role not in ["instructor", "admin"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"User {course_data.instructor_id} is not an instructor"
                    )

        # Check for duplicate title if title is being updated
        if course_data.title:
            async with async_db.get_session() as session:
                existing_result = await session.execute(
                    select(Course).where(
                        and_(
                            Course.title == course_data.title,
                            Course.instructor_id == (course_data.instructor_id or course.instructor_id),
                            Course.course_id != course_id
                        )
                    )
                )
                existing_course = existing_result.scalar_one_or_none()

                if existing_course:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Course '{course_data.title}' already exists for this instructor"
                    )

        # Update course
        update_data = course_data.model_dump(exclude_unset=True)
        updated_course = await course_crud.update(course_id, update_data)

        if not updated_course:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        api_logger.info(f"✅ Updated course {course_id}")

        # Get updated instructor details
        async with async_db.get_session() as session:
            instructor_result = await session.execute(
                select(User.username, User.email)
                .where(User.user_id == updated_course.instructor_id)
            )
            instructor_data = instructor_result.first()

        return CourseResponse(
            course_id=updated_course.course_id,
            title=updated_course.title,
            description=updated_course.description,
            instructor_id=updated_course.instructor_id,
            instructor_name=instructor_data.username if instructor_data else None,
            instructor_email=instructor_data.email if instructor_data else None,
            created_at=updated_course.created_at.isoformat(),
            updated_at=updated_course.updated_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to update course {course_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{course_id}")
async def delete_course(course_id: str) -> Dict[str, str]:
    """Delete course by ID."""
    try:
        # Check if course exists
        async with async_db.get_session() as session:
            result = await session.execute(
                select(Course).where(Course.course_id == course_id)
            )
            course = result.scalar_one_or_none()

            if not course:
                raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        # Delete course (cascade will handle related records)
        deleted = await course_crud.delete(course_id)

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        api_logger.info(f"✅ Deleted course {course_id}")

        return {"message": f"Course {course_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to delete course {course_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=CourseListResponse)
async def list_courses(
    instructor_id: Optional[str] = Query(None, description="Filter by instructor ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
) -> CourseListResponse:
    """List courses with pagination and optional instructor filtering."""
    try:
        if instructor_id:
            # Filter by specific instructor
            courses, total_count = await course_crud.get_courses_by_instructor(
                instructor_id=instructor_id,
                page=page,
                limit=limit
            )
        else:
            # Get all courses with pagination
            async with async_db.get_session() as session:
                # Get total count
                count_result = await session.execute(
                    select(func.count(Course.course_id))
                )
                total_count = count_result.scalar()

                # Get paginated results with instructor info
                result = await session.execute(
                    select(Course, User.username, User.email)
                    .join(User, Course.instructor_id == User.user_id)
                    .order_by(desc(Course.updated_at))
                    .limit(limit)
                    .offset((page - 1) * limit)
                )

                # Store courses with instructor info
                courses_with_instructors = []
                for row in result.fetchall():
                    course, username, email = row
                    courses_with_instructors.append({
                        'course': course,
                        'instructor_name': username,
                        'instructor_email': email
                    })

        course_responses = []
        if instructor_id:
            # Handle filtered by instructor case
            for course_data in courses:
                course = course_data['course']
                course_responses.append(CourseResponse(
                    course_id=course.course_id,
                    title=course.title,
                    description=course.description,
                    instructor_id=course.instructor_id,
                    instructor_name=course_data['instructor_name'],
                    instructor_email=course_data['instructor_email'],
                    created_at=course.created_at.isoformat(),
                    updated_at=course.updated_at.isoformat()
                ))
        else:
            # Handle get all courses case (instructor info already included)
            for course_data in courses_with_instructors:
                course = course_data['course']
                course_responses.append(CourseResponse(
                    course_id=course.course_id,
                    title=course.title,
                    description=course.description,
                    instructor_id=course.instructor_id,
                    instructor_name=course_data['instructor_name'],
                    instructor_email=course_data['instructor_email'],
                    created_at=course.created_at.isoformat(),
                    updated_at=course.updated_at.isoformat()
                ))

        return CourseListResponse(
            courses=course_responses,
            total_count=total_count,
            page=page,
            limit=limit
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to list courses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/instructor/{instructor_id}", response_model=CourseListResponse)
async def get_instructor_courses(
    instructor_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
) -> CourseListResponse:
    """Get courses for a specific instructor."""
    try:
        courses, total_count = await course_crud.get_courses_by_instructor(
            instructor_id=instructor_id,
            page=page,
            limit=limit,
            include_stats=True
        )

        course_responses = []
        if instructor_id:
            # Handle filtered by instructor case
            for course_data in courses:
                course = course_data['course']
                course_responses.append(CourseResponse(
                    course_id=course.course_id,
                    title=course.title,
                    description=course.description,
                    instructor_id=course.instructor_id,
                    instructor_name=course_data['instructor_name'],
                    instructor_email=course_data['instructor_email'],
                    created_at=course.created_at.isoformat(),
                    updated_at=course.updated_at.isoformat()
                ))
        else:
            # Handle get all courses case (instructor info already included)
            for course_data in courses_with_instructors:
                course = course_data['course']
                course_responses.append(CourseResponse(
                    course_id=course.course_id,
                    title=course.title,
                    description=course.description,
                    instructor_id=course.instructor_id,
                    instructor_name=course_data['instructor_name'],
                    instructor_email=course_data['instructor_email'],
                    created_at=course.created_at.isoformat(),
                    updated_at=course.updated_at.isoformat()
                ))

        return CourseListResponse(
            courses=course_responses,
            total_count=total_count,
            page=page,
            limit=limit
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get instructor courses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/{query}", response_model=CourseListResponse)
async def search_courses_endpoint(
    query: str,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
) -> CourseListResponse:
    """Search courses by title or description."""
    try:
        courses, total_count = await course_crud.search_courses(
            query=query,
            page=page,
            limit=limit
        )

        course_responses = []
        if instructor_id:
            # Handle filtered by instructor case
            for course_data in courses:
                course = course_data['course']
                course_responses.append(CourseResponse(
                    course_id=course.course_id,
                    title=course.title,
                    description=course.description,
                    instructor_id=course.instructor_id,
                    instructor_name=course_data['instructor_name'],
                    instructor_email=course_data['instructor_email'],
                    created_at=course.created_at.isoformat(),
                    updated_at=course.updated_at.isoformat()
                ))
        else:
            # Handle get all courses case (instructor info already included)
            for course_data in courses_with_instructors:
                course = course_data['course']
                course_responses.append(CourseResponse(
                    course_id=course.course_id,
                    title=course.title,
                    description=course.description,
                    instructor_id=course.instructor_id,
                    instructor_name=course_data['instructor_name'],
                    instructor_email=course_data['instructor_email'],
                    created_at=course.created_at.isoformat(),
                    updated_at=course.updated_at.isoformat()
                ))

        return CourseListResponse(
            courses=course_responses,
            total_count=total_count,
            page=page,
            limit=limit
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to search courses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{course_id}/stats", response_model=CourseStatsResponse)
async def get_course_statistics(course_id: str) -> CourseStatsResponse:
    """Get detailed statistics for a specific course."""
    try:
        course_details = await course_crud.get_course_with_details(course_id)

        if not course_details:
            raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

        stats = course_details["stats"]

        return CourseStatsResponse(
            course_id=course_id,
            documents_count=stats["documents_count"],
            chatrooms_count=stats["chatrooms_count"],
            user_contexts_count=stats["user_contexts_count"],
            total_file_size=stats["total_file_size"],
            processed_documents=stats["processed_documents"],
            pending_documents=stats["pending_documents"]
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get course statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/popular/top", response_model=List[Dict[str, Any]])
async def get_popular_courses(limit: int = Query(10, ge=1, le=50)) -> List[Dict[str, Any]]:
    """Get popular courses based on usage."""
    try:
        popular_courses = await course_crud.get_popular_courses(limit=limit)
        return popular_courses

    except Exception as e:
        api_logger.error(f"Failed to get popular courses: {e}")
        raise HTTPException(status_code=500, detail=str(e))