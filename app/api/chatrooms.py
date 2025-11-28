"""
Chatroom Management API
CRUD operations for chatroom management with proper database integration
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import time

from ..core.database import async_db, AsyncCRUD
from ..schemas.db_models import Chatroom, User, Course, UserContext, Message, Response
import uuid
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

# Pydantic models for messages and responses
class MessageResponse(BaseModel):
    message_id: str
    chatroom_id: str
    user_id: str
    message_text: Optional[str] = None
    created_at: str
    responses: List["ResponseResponse"] = []

class ResponseResponse(BaseModel):
    response_id: str
    message_id: str
    chatroom_id: str
    user_id: str
    response_text: str
    model_used: str
    response_type: str
    source_type: str
    cache_hit: bool
    cache_similarity_score: Optional[float] = None
    personalized: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    created_at: str
    updated_at: str

class ChatroomMessagesResponse(BaseModel):
    chatroom_id: str
    messages: List[MessageResponse]
    total_messages: int
    total_responses: int

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

# Helper function for welcome message
async def create_welcome_message(chatroom_id: str, user_id: str, course_title: str, user_name: str = None):
    """Create automatic AI response welcome message when chatroom is created (no user message needed)."""
    try:
        async with async_db.get_session() as session:
            # Create a dummy system message for the response to link to
            system_message = Message(
                message_id=str(uuid.uuid4()),
                chatroom_id=chatroom_id,
                user_id=user_id,
                message_text=None
            )
            session.add(system_message)
            await session.flush()

            # Create AI response welcome message
            welcome_response_text = f"""👋 Halo {user_name or 'Pengguna'}!

Selamat datang di chatroom untuk course *{course_title}*! 🚀

Saya adalah AI Tutor yang siap membantu kamu:
- 📚 Menjelaskan materi yang sulit dipahami
- 💡 Memberikan contoh dan ilustrasi
- 🎯 Membantu mengerjakan latihan soal
- ❓ Menjawab pertanyaan seputar course ini
- 📝 Membuat rangkuman materi

Jangan ragu bertanya ya! Saya akan berusaha menjawab dengan cara yang mudah kamu pahami.

Apa yang ingin kita pelajari hari ini? 😊"""

            response = Response(
                response_id=str(uuid.uuid4()),
                message_id=system_message.message_id,
                chatroom_id=chatroom_id,
                user_id=user_id,
                response_text=welcome_response_text,
                model_used="system",
                response_type="cache_hit_raw",
                source_type="redis_cache",
                cache_hit=False,
                personalized=False,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                latency_ms=0.0
            )
            session.add(response)
            await session.commit()

            api_logger.info(f"✅ Created welcome message for chatroom {chatroom_id}")
            return message.message_id

    except Exception as e:
        api_logger.error(f"❌ Failed to create welcome message: {e}")
        return None

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

        # Multiple chatrooms allowed for same user-course combination
        # Always create new chatroom regardless of existing ones

        # Create chatroom
        chatroom_dict = chatroom_data.model_dump()
        chatroom = await chatroom_crud.create(chatroom_dict)

        api_logger.info(f"✅ Created chatroom {chatroom.chatroom_id}")

        # Create user context and welcome message
        async with async_db.get_session() as session:
            try:
                # Create user context for this course if it doesn't exist
                await UserContext.aget_or_create(
                    session,
                    chatroom_data.user_id,
                    chatroom_data.course_id,
                    initial_context="New user - ready to start learning!"
                )

                # Get user name and course title for personalized welcome
                user_result = await session.execute(
                    select(User).where(User.user_id == chatroom_data.user_id)
                )
                user = user_result.scalar_one_or_none()
                user_name = user.username if user else None

                course_result = await session.execute(
                    select(Course).where(Course.course_id == chatroom_data.course_id)
                )
                course = course_result.scalar_one_or_none()
                course_title = course.title if course else "kursus"

                # Create automatic welcome message
                await create_welcome_message(
                    chatroom.chatroom_id,
                    chatroom_data.user_id,
                    course_title,
                    user_name
                )
            except Exception as e:
                api_logger.warning(f"Failed to create user context or welcome message: {e}")

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

@router.get("/{chatroom_id}/messages", response_model=ChatroomMessagesResponse)
async def get_chatroom_messages(
    chatroom_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=200, description="Messages per page"),
    include_responses: bool = Query(True, description="Include responses for each message")
) -> ChatroomMessagesResponse:
    """Get all messages and their responses for a specific chatroom."""
    try:
        # Verify chatroom exists
        async with async_db.get_session() as session:
            chatroom_result = await session.execute(
                select(Chatroom).where(Chatroom.chatroom_id == chatroom_id)
            )
            chatroom = chatroom_result.scalar_one_or_none()

            if not chatroom:
                raise HTTPException(status_code=404, detail=f"Chatroom {chatroom_id} not found")

        # Get messages with pagination
        async with async_db.get_session() as session:
            # Get total message count
            message_count_result = await session.execute(
                select(Message.message_id).where(Message.chatroom_id == chatroom_id)
            )
            total_messages = len(message_count_result.scalars().all())

            # Get paginated messages
            messages_query = (
                select(Message)
                .where(Message.chatroom_id == chatroom_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .offset((page - 1) * limit)
            )

            messages_result = await session.execute(messages_query)
            messages = messages_result.scalars().all()

            # Get total response count for this chatroom
            if include_responses:
                response_count_result = await session.execute(
                    select(Response.response_id).where(Response.chatroom_id == chatroom_id)
                )
                total_responses = len(response_count_result.scalars().all())
            else:
                total_responses = 0

            # Build response with optional responses
            message_responses = []
            for message in messages:
                message_response = MessageResponse(
                    message_id=message.message_id,
                    chatroom_id=message.chatroom_id,
                    user_id=message.user_id,
                    message_text=message.message_text,
                    created_at=message.created_at.isoformat(),
                    responses=[]
                )

                # Get responses for this message if requested
                if include_responses:
                    responses_query = (
                        select(Response)
                        .where(Response.message_id == message.message_id)
                        .order_by(Response.created_at.asc())
                    )

                    responses_result = await session.execute(responses_query)
                    responses = responses_result.scalars().all()

                    for response in responses:
                        response_response = ResponseResponse(
                            response_id=response.response_id,
                            message_id=response.message_id,
                            chatroom_id=response.chatroom_id,
                            user_id=response.user_id,
                            response_text=response.response_text,
                            model_used=response.model_used,
                            response_type=response.response_type,
                            source_type=response.source_type,
                            cache_hit=response.cache_hit,
                            cache_similarity_score=response.cache_similarity_score,
                            personalized=response.personalized,
                            input_tokens=response.input_tokens,
                            output_tokens=response.output_tokens,
                            total_tokens=response.total_tokens,
                            cost_usd=response.cost_usd,
                            latency_ms=response.latency_ms,
                            created_at=response.created_at.isoformat(),
                            updated_at=response.updated_at.isoformat()
                        )
                        message_response.responses.append(response_response)

                message_responses.append(message_response)

        api_logger.info(f"Retrieved {len(messages)} messages for chatroom {chatroom_id} (page {page})")

        return ChatroomMessagesResponse(
            chatroom_id=chatroom_id,
            messages=message_responses,
            total_messages=total_messages,
            total_responses=total_responses
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get messages for chatroom {chatroom_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))