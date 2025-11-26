"""
Chatroom Management API
CRUD operations for chatroom management with proper database integration
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import time

from ..core.database import async_db, AsyncCRUD
from ..schemas.db_models import Chatroom, User, Course, UserContext
from ..core.logger import api_logger

router = APIRouter(prefix="/chatrooms", tags=["chatrooms"])

# Pydantic models for request/response
class ChatroomCreate(BaseModel):
    room_name: str = Field(..., min_length=1, max_length=255, description="Chatroom name")
    description: Optional[str] = Field(None, max_length=1000, description="Chatroom description")
    user_id: str = Field(..., description="User ID who owns the chatroom")
    course_id: str = Field(..., description="Course ID for the chatroom")
    max_messages: Optional[int] = Field(1000, ge=1, le=10000, description="Maximum messages per chatroom")

class ChatroomUpdate(BaseModel):
    room_name: Optional[str] = Field(None, min_length=1, max_length=255, description="Updated chatroom name")
    description: Optional[str] = Field(None, max_length=1000, description="Updated chatroom description")
    is_active: Optional[bool] = Field(None, description="Activate/deactivate chatroom")
    max_messages: Optional[int] = Field(None, ge=1, le=10000, description="Maximum messages per chatroom")

class ChatroomResponse(BaseModel):
    chatroom_id: str
    user_id: str
    course_id: str
    room_name: str
    description: Optional[str]
    is_active: bool
    max_messages: int
    created_at: str
    updated_at: str
    message_count: Optional[int] = None

class ChatroomListResponse(BaseModel):
    chatrooms: List[ChatroomResponse]
    total_count: int
    page: int
    limit: int

# CRUD operations using AsyncCRUD base class
class ChatroomCRUD(AsyncCRUD):
    def __init__(self):
        super().__init__(Chatroom)

    async def get_by_user_course(self, user_id: str, course_id: str) -> Optional[Chatroom]:
        """Get chatroom by user and course combination."""
        async with async_db.get_session() as session:
            result = await session.execute(
                select(Chatroom).where(
                    and_(
                        Chatroom.user_id == user_id,
                        Chatroom.course_id == course_id,
                        Chatroom.is_active == True
                    )
                )
            )
            return result.scalar_one_or_none()

    async def get_user_chatrooms(
        self,
        user_id: str,
        course_id: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
        include_inactive: bool = False
    ) -> tuple[List[Chatroom], int]:
        """Get paginated chatrooms for a user."""
        async with async_db.get_session() as session:
            # Build base query
            base_conditions = [Chatroom.user_id == user_id]

            if course_id:
                base_conditions.append(Chatroom.course_id == course_id)

            if not include_inactive:
                base_conditions.append(Chatroom.is_active == True)

            # Get total count
            count_result = await session.execute(
                select(Chatroom.chatroom_id).where(and_(*base_conditions))
            )
            total_count = len(count_result.scalars().all())

            # Get paginated results
            result = await session.execute(
                select(Chatroom, User.username, Course.title)
                .join(User, Chatroom.user_id == User.user_id)
                .join(Course, Chatroom.course_id == Course.course_id)
                .where(and_(*base_conditions))
                .order_by(desc(Chatroom.updated_at))
                .limit(limit)
                .offset((page - 1) * limit)
            )

            chatrooms = []
            for row in result.fetchall():
                chatroom, username, course_title = row
                chatrooms.append(chatroom)

            return chatrooms, total_count

    async def get_chatroom_with_details(self, chatroom_id: str) -> Optional[Dict[str, Any]]:
        """Get chatroom with user and course details."""
        async with async_db.get_session() as session:
            result = await session.execute(
                select(Chatroom, User.username, User.email, Course.title, Course.description)
                .join(User, Chatroom.user_id == User.user_id)
                .join(Course, Chatroom.course_id == Course.course_id)
                .where(Chatroom.chatroom_id == chatroom_id)
            )

            row = result.first()
            if not row:
                return None

            chatroom, username, email, course_title, course_description = row

            return {
                "chatroom": chatroom,
                "user_info": {
                    "username": username,
                    "email": email
                },
                "course_info": {
                    "title": course_title,
                    "description": course_description
                }
            }

# Initialize CRUD
chatroom_crud = ChatroomCRUD()

@router.post("/", response_model=ChatroomResponse, status_code=201)
async def create_chatroom(chatroom_data: ChatroomCreate) -> ChatroomResponse:
    """Create a new chatroom."""
    try:
        api_logger.info(f"Creating chatroom: {chatroom_data.room_name} for user {chatroom_data.user_id}")

        # Check if user exists
        async with async_db.get_session() as session:
            user_result = await session.execute(
                select(User).where(User.user_id == chatroom_data.user_id)
            )
            user = user_result.scalar_one_or_none()

            if not user:
                raise HTTPException(status_code=404, detail=f"User {chatroom_data.user_id} not found")

        # Check if course exists
        async with async_db.get_session() as session:
            course_result = await session.execute(
                select(Course).where(Course.course_id == chatroom_data.course_id)
            )
            course = course_result.scalar_one_or_none()

            if not course:
                raise HTTPException(status_code=404, detail=f"Course {chatroom_data.course_id} not found")

        # Check if chatroom already exists for this user-course combination
        existing_chatroom = await chatroom_crud.get_by_user_course(
            chatroom_data.user_id,
            chatroom_data.course_id
        )

        if existing_chatroom:
            api_logger.info(f"Returning existing chatroom for user {chatroom_data.user_id} in course {chatroom_data.course_id}")
            return ChatroomResponse(
                chatroom_id=existing_chatroom.chatroom_id,
                user_id=existing_chatroom.user_id,
                course_id=existing_chatroom.course_id,
                room_name=existing_chatroom.room_name,
                description=existing_chatroom.description,
                is_active=existing_chatroom.is_active,
                max_messages=existing_chatroom.max_messages,
                created_at=existing_chatroom.created_at.isoformat(),
                updated_at=existing_chatroom.updated_at.isoformat()
            )

        # Create chatroom
        chatroom_dict = chatroom_data.model_dump()
        chatroom = await chatroom_crud.create(chatroom_dict)

        api_logger.info(f"✅ Created chatroom {chatroom.chatroom_id}")

        # Create user context for this course if it doesn't exist
        async with async_db.get_session() as session:
            try:
                await UserContext.aget_or_create(
                    session,
                    chatroom_data.user_id,
                    chatroom_data.course_id,
                    initial_context="New user - ready to start learning!"
                )
            except Exception as e:
                api_logger.warning(f"Failed to create user context: {e}")

        return ChatroomResponse(
            chatroom_id=chatroom.chatroom_id,
            user_id=chatroom.user_id,
            course_id=chatroom.course_id,
            room_name=chatroom.room_name,
            description=chatroom.description,
            is_active=chatroom.is_active,
            max_messages=chatroom.max_messages,
            created_at=chatroom.created_at.isoformat(),
            updated_at=chatroom.updated_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to create chatroom: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{chatroom_id}", response_model=Dict[str, Any])
async def get_chatroom(chatroom_id: str) -> Dict[str, Any]:
    """Get chatroom by ID with user and course details."""
    try:
        chatroom_details = await chatroom_crud.get_chatroom_with_details(chatroom_id)

        if not chatroom_details:
            raise HTTPException(status_code=404, detail=f"Chatroom {chatroom_id} not found")

        chatroom = chatroom_details["chatroom"]

        return {
            "chatroom_id": chatroom.chatroom_id,
            "user_id": chatroom.user_id,
            "course_id": chatroom.course_id,
            "room_name": chatroom.room_name,
            "description": chatroom.description,
            "is_active": chatroom.is_active,
            "max_messages": chatroom.max_messages,
            "created_at": chatroom.created_at.isoformat(),
            "updated_at": chatroom.updated_at.isoformat(),
            "user_info": chatroom_details["user_info"],
            "course_info": chatroom_details["course_info"]
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get chatroom {chatroom_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{chatroom_id}", response_model=ChatroomResponse)
async def update_chatroom(chatroom_id: str, chatroom_data: ChatroomUpdate) -> ChatroomResponse:
    """Update chatroom by ID."""
    try:
        # Check if chatroom exists
        async with async_db.get_session() as session:
            result = await session.execute(
                select(Chatroom).where(Chatroom.chatroom_id == chatroom_id)
            )
            chatroom = result.scalar_one_or_none()

            if not chatroom:
                raise HTTPException(status_code=404, detail=f"Chatroom {chatroom_id} not found")

        # Update chatroom
        update_data = chatroom_data.model_dump(exclude_unset=True)
        updated_chatroom = await chatroom_crud.update(chatroom_id, update_data)

        if not updated_chatroom:
            raise HTTPException(status_code=404, detail=f"Chatroom {chatroom_id} not found")

        api_logger.info(f"✅ Updated chatroom {chatroom_id}")

        return ChatroomResponse(
            chatroom_id=updated_chatroom.chatroom_id,
            user_id=updated_chatroom.user_id,
            course_id=updated_chatroom.course_id,
            room_name=updated_chatroom.room_name,
            description=updated_chatroom.description,
            is_active=updated_chatroom.is_active,
            max_messages=updated_chatroom.max_messages,
            created_at=updated_chatroom.created_at.isoformat(),
            updated_at=updated_chatroom.updated_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to update chatroom {chatroom_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{chatroom_id}")
async def delete_chatroom(chatroom_id: str) -> Dict[str, str]:
    """Delete chatroom by ID."""
    try:
        # Check if chatroom exists
        async with async_db.get_session() as session:
            result = await session.execute(
                select(Chatroom).where(Chatroom.chatroom_id == chatroom_id)
            )
            chatroom = result.scalar_one_or_none()

            if not chatroom:
                raise HTTPException(status_code=404, detail=f"Chatroom {chatroom_id} not found")

        # Delete chatroom
        deleted = await chatroom_crud.delete(chatroom_id)

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Chatroom {chatroom_id} not found")

        api_logger.info(f"✅ Deleted chatroom {chatroom_id}")

        return {"message": f"Chatroom {chatroom_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to delete chatroom {chatroom_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=ChatroomListResponse)
async def list_chatrooms(
    user_id: str = Query(..., description="User ID (required)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
) -> ChatroomListResponse:
    """List chatrooms with pagination."""
    try:
        chatrooms, total_count = await chatroom_crud.get_user_chatrooms(
            user_id=user_id,
            course_id=None,
            page=page,
            limit=limit,
            include_inactive=True
        )

        chatroom_responses = []
        for chatroom in chatrooms:
            chatroom_responses.append(ChatroomResponse(
                chatroom_id=chatroom.chatroom_id,
                user_id=chatroom.user_id,
                course_id=chatroom.course_id,
                room_name=chatroom.room_name,
                description=chatroom.description,
                is_active=chatroom.is_active,
                max_messages=chatroom.max_messages,
                created_at=chatroom.created_at.isoformat(),
                updated_at=chatroom.updated_at.isoformat()
            ))

        return ChatroomListResponse(
            chatrooms=chatroom_responses,
            total_count=total_count,
            page=page,
            limit=limit
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to list chatrooms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}/course/{course_id}", response_model=ChatroomResponse)
async def get_user_course_chatroom(user_id: str, course_id: str) -> ChatroomResponse:
    """Get chatroom for specific user and course combination."""
    try:
        chatroom = await chatroom_crud.get_by_user_course(user_id, course_id)

        if not chatroom:
            raise HTTPException(
                status_code=404,
                detail=f"No active chatroom found for user {user_id} in course {course_id}"
            )

        return ChatroomResponse(
            chatroom_id=chatroom.chatroom_id,
            user_id=chatroom.user_id,
            course_id=chatroom.course_id,
            room_name=chatroom.room_name,
            description=chatroom.description,
            is_active=chatroom.is_active,
            max_messages=chatroom.max_messages,
            created_at=chatroom.created_at.isoformat(),
            updated_at=chatroom.updated_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get user-course chatroom: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{chatroom_id}/reactivate")
async def reactivate_chatroom(chatroom_id: str) -> ChatroomResponse:
    """Reactivate a deactivated chatroom."""
    try:
        updated_chatroom = await chatroom_crud.update(chatroom_id, {"is_active": True})

        if not updated_chatroom:
            raise HTTPException(status_code=404, detail=f"Chatroom {chatroom_id} not found")

        api_logger.info(f"✅ Reactivated chatroom {chatroom_id}")

        return ChatroomResponse(
            chatroom_id=updated_chatroom.chatroom_id,
            user_id=updated_chatroom.user_id,
            course_id=updated_chatroom.course_id,
            room_name=updated_chatroom.room_name,
            description=updated_chatroom.description,
            is_active=updated_chatroom.is_active,
            max_messages=updated_chatroom.max_messages,
            created_at=updated_chatroom.created_at.isoformat(),
            updated_at=updated_chatroom.updated_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to reactivate chatroom {chatroom_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))