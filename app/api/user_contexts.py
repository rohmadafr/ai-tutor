"""
UserContext CRUD API with Template Support
Manages user personalization contexts for chatbot interactions
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy import select, and_, or_
import time
import datetime

from ..core.database import async_db
from ..core.logger import api_logger
from ..schemas.db_models import UserContext, User, Course
from ..core.exceptions import TutorServiceException

router = APIRouter(prefix="/user-contexts", tags=["user-contexts"])

# Constants for validation
MAX_CONTEXT_LENGTH = 2000  # Maximum characters to stay within token limits

# Pydantic models
class UserContextCreate(BaseModel):
    """User context creation model"""
    user_id: str = Field(..., description="User ID")
    course_id: str = Field(..., description="Course ID")
    user_context: str = Field(..., min_length=1, max_length=MAX_CONTEXT_LENGTH, description="User context text")
    is_template: bool = Field(False, description="Whether this is a template context")
    template_name: Optional[str] = Field(None, description="Template name (if is_template=True)")

class UserContextUpdate(BaseModel):
    """User context update model"""
    user_context: str = Field(..., min_length=1, max_length=MAX_CONTEXT_LENGTH, description="Updated user context text")
    template_name: Optional[str] = Field(None, description="Updated template name")

class UserContextResponse(BaseModel):
    """User context response model"""
    context_id: str
    user_id: str
    course_id: str
    user_context: str
    is_template: bool = False
    template_name: Optional[str] = None
    created_at: str
    updated_at: str
    last_used_at: str

    class Config:
        from_attributes = True

class UserContextTemplate(BaseModel):
    """User context template model"""
    template_id: str
    template_name: str
    user_context: str
    description: str
    category: str = "general"  # general, beginner, advanced, etc.

# Default templates
DEFAULT_TEMPLATES = [
    {
        "template_id": "template_beginner",
        "template_name": "Pemula Dasar",
        "user_context": """Preferensi: Saya lebih suka penjelasan yang sederhana, langkah demi langkah dengan contoh konkret. Saya butuh waktu lebih lama untuk memahami konsep baru.
Goals: Ingin memahami fundamental konsep, lulus ujian dengan nilai baik, dan bisa aplikasikan ilmu dalam tugas-tugas dasar.
Learning style: Visual dan praktis, lebih suka lihat demonstrasi dan praktek langsung daripada teori panjang.
Informasi tambahan: Mohon berikan contoh sederhana dan analogi yang mudah dipahami. Hindari terminologi yang terlalu teknis tanpa penjelasan.""",
        "description": "Template untuk pemula yang butuh penjelasan sederhana dan langkah demi langkah",
        "category": "beginner"
    },
    {
        "template_id": "template_intermediate",
        "template_name": "Menengah Praktis",
        "user_context": """Preferensi: Saya suka penjelasan yang seimbang antara teori dan praktik. Berikan saya contoh nyata dan kasus-kasus aplikasi yang relevan.
Goals: Mengembangkan pemahaman mendalam, meningkatkan kemampuan problem solving, dan siap untuk tantangan tingkat lanjut.
Learning style: Kinestetik dan logis, lebih suka coba-coba langsung dan analisis pola serta hubungan konsep.
Informasi tambahan: Saya terbuka untuk diskusi mendalam dan tantangan. Berikan saya kasus studi atau mini-project untuk praktik.""",
        "description": "Template untuk learner tingkat menengah yang siap dengan tantangan",
        "category": "intermediate"
    },
    {
        "template_id": "template_advanced",
        "template_name": "Lanjutan Eksploratif",
        "user_context": """Preferensi: Saya suka diskusi mendalam, analisis kompleks, dan eksplorasi konsep yang advanced. Berikan saya multiple perspectives dan critical thinking challenges.
Goals: Master topik secara komprehensif, kembangkan insight baru, dan siap untuk implementasi tingkat profesional atau riset.
Learning style: Analitik dan eksploratif, lebih suka case studies kompleks, research papers, dan problem solving tingkat tinggi.
Informasi tambahan: Jangan ragu memberikan reading materials lanjutan, research terkini, atau topik controversial untuk diskusi. Saya siap dengan challenge.""",
        "description": "Template untuk advanced learners yang siap dengan eksplorasi mendalam",
        "category": "advanced"
    }
]

class AsyncCRUD:
    """Async CRUD operations for UserContext"""

    @staticmethod
    async def get_by_id(context_id: str) -> Optional[UserContext]:
        """Get user context by ID - returns detached object"""
        async with async_db.get_session() as db:
            result = await db.execute(select(UserContext).where(UserContext.context_id == context_id))
            return result.scalar_one_or_none()

    @staticmethod
    async def get_user_course_context(user_id: str, course_id: str) -> Optional[UserContext]:
        """Get user context for specific user and course"""
        async with async_db.get_session() as db:
            result = await db.execute(
                select(UserContext).where(
                    and_(
                        UserContext.user_id == user_id,
                        UserContext.course_id == course_id
                    )
                )
            )
            return result.scalar_one_or_none()

    @staticmethod
    async def get_user_contexts(
        user_id: Optional[str] = None,
        course_id: Optional[str] = None,
        is_template: Optional[bool] = None,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[UserContext]:
        """Get user contexts with filters"""
        async with async_db.get_session() as db:
            query = select(UserContext)

            # Apply filters
            if user_id:
                query = query.where(UserContext.user_id == user_id)
            if course_id:
                query = query.where(UserContext.course_id == course_id)
            if is_template is not None:
                # Note: is_template would need to be added to UserContext model
                # For now, we'll filter by template_name presence
                if is_template:
                    query = query.where(UserContext.user_context.like("%template_id:%"))
                else:
                    query = query.where(~UserContext.user_context.like("%template_id:%"))

            # Add pagination
            query = query.offset(offset).limit(limit)

            result = await db.execute(query)
            return result.scalars().all()

    @staticmethod
    async def create(user_context_data: dict) -> UserContext:
        """Create new user context"""
        async with async_db.get_session() as db:
            # Check if context already exists for this user-course combination
            existing = await AsyncCRUD.get_user_course_context(
                user_context_data["user_id"],
                user_context_data["course_id"]
            )

            if existing:
                raise TutorServiceException(
                    "User context already exists for this user and course",
                    error_code="CONTEXT_EXISTS"
                )

            # Validate user and course exist
            user_result = await db.execute(select(User).where(User.user_id == user_context_data["user_id"]))
            if not user_result.scalar_one_or_none():
                raise HTTPException(status_code=404, detail="User not found")

            course_result = await db.execute(select(Course).where(Course.course_id == user_context_data["course_id"]))
            if not course_result.scalar_one_or_none():
                raise HTTPException(status_code=404, detail="Course not found")

            # Create new context
            new_context = UserContext(**user_context_data)
            db.add(new_context)
            await db.flush()
            await db.refresh(new_context)

            api_logger.info(f"Created user context: {new_context.context_id} for user {user_context_data['user_id']}")
            return new_context

    @staticmethod
    async def update(context_id: str, update_data: dict) -> Optional[UserContext]:
        """Update user context"""
        async with async_db.get_session() as db:
            # Fetch object in the same session
            result = await db.execute(select(UserContext).where(UserContext.context_id == context_id))
            context = result.scalar_one_or_none()

            if not context:
                return None

            for key, value in update_data.items():
                setattr(context, key, value)

            context.updated_at = datetime.datetime.utcnow()
            await db.commit()
            await db.refresh(context)

            api_logger.info(f"Updated user context: {context_id}")
            return context

    @staticmethod
    async def delete(context_id: str) -> bool:
        """Delete user context"""
        async with async_db.get_session() as db:
            context = await AsyncCRUD.get_by_id(context_id)
            if not context:
                return False

            await db.delete(context)
            await db.commit()

            api_logger.info(f"Deleted user context: {context_id}")
            return True

# API Endpoints
@router.get("/", response_model=List[UserContextResponse])
async def get_user_contexts(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    course_id: Optional[str] = Query(None, description="Filter by course ID"),
    is_template: Optional[bool] = Query(None, description="Filter templates only"),
    limit: int = Query(100, ge=1, le=1000, description="Number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """Get user contexts with optional filters"""
    try:
        contexts = await AsyncCRUD.get_user_contexts(
            user_id=user_id,
            course_id=course_id,
            is_template=is_template,
            limit=limit,
            offset=offset
        )

        return [
            UserContextResponse(
                context_id=ctx.context_id,
                user_id=ctx.user_id,
                course_id=ctx.course_id,
                user_context=ctx.user_context,
                created_at=ctx.created_at.isoformat(),
                updated_at=ctx.updated_at.isoformat(),
                last_used_at=ctx.last_used_at.isoformat()
            ) for ctx in contexts
        ]
    except Exception as e:
        api_logger.error(f"Failed to get user contexts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user contexts")

@router.get("/templates", response_model=List[UserContextTemplate])
async def get_user_context_templates():
    """Get available user context templates"""
    try:
        return [
            UserContextTemplate(**template)
            for template in DEFAULT_TEMPLATES
        ]
    except Exception as e:
        api_logger.error(f"Failed to get user context templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve templates")

@router.get("/{context_id}", response_model=UserContextResponse)
async def get_user_context(context_id: str):
    """Get specific user context by ID"""
    try:
        context = await AsyncCRUD.get_by_id(context_id)
        if not context:
            raise HTTPException(status_code=404, detail="User context not found")

        return UserContextResponse(
            context_id=context.context_id,
            user_id=context.user_id,
            course_id=context.course_id,
            user_context=context.user_context,
            created_at=context.created_at.isoformat(),
            updated_at=context.updated_at.isoformat(),
            last_used_at=context.last_used_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get user context {context_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user context")

@router.get("/user/{user_id}/course/{course_id}", response_model=UserContextResponse)
async def get_user_course_context(user_id: str, course_id: str):
    """Get user context for specific user and course"""
    try:
        context = await AsyncCRUD.get_user_course_context(user_id, course_id)
        if not context:
            raise HTTPException(status_code=404, detail="User context not found for this user and course")

        return UserContextResponse(
            context_id=context.context_id,
            user_id=context.user_id,
            course_id=context.course_id,
            user_context=context.user_context,
            created_at=context.created_at.isoformat(),
            updated_at=context.updated_at.isoformat(),
            last_used_at=context.last_used_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get user course context: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user context")

@router.post("/", response_model=UserContextResponse)
async def create_user_context(context_data: UserContextCreate):
    """Create new user context"""
    try:
        # Convert to dict and add timestamps
        context_dict = context_data.dict()

        # Check if this is a template creation
        if context_data.is_template:
            # Add template identifier to context text for tracking
            if not context_dict["user_context"].startswith("template_id:"):
                context_dict["user_context"] = f"template_id:{context_data.template_name or 'custom'}\n{context_dict['user_context']}"

        new_context = await AsyncCRUD.create(context_dict)

        return UserContextResponse(
            context_id=new_context.context_id,
            user_id=new_context.user_id,
            course_id=new_context.course_id,
            user_context=new_context.user_context,
            created_at=new_context.created_at.isoformat(),
            updated_at=new_context.updated_at.isoformat(),
            last_used_at=new_context.last_used_at.isoformat()
        )
    except TutorServiceException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to create user context: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user context")

@router.post("/from-template", response_model=UserContextResponse)
async def create_user_context_from_template(
    user_id: str,
    course_id: str,
    template_id: str
):
    """Create user context from template"""
    try:
        # Find template
        template = next((t for t in DEFAULT_TEMPLATES if t["template_id"] == template_id), None)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        # Create context data
        context_data = {
            "user_id": user_id,
            "course_id": course_id,
            "user_context": template["user_context"]
        }

        new_context = await AsyncCRUD.create(context_data)

        return UserContextResponse(
            context_id=new_context.context_id,
            user_id=new_context.user_id,
            course_id=new_context.course_id,
            user_context=new_context.user_context,
            created_at=new_context.created_at.isoformat(),
            updated_at=new_context.updated_at.isoformat(),
            last_used_at=new_context.last_used_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to create user context from template: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user context from template")

@router.put("/{context_id}", response_model=UserContextResponse)
async def update_user_context(context_id: str, update_data: UserContextUpdate):
    """Update user context"""
    try:
        update_dict = update_data.dict()

        updated_context = await AsyncCRUD.update(context_id, update_dict)
        if not updated_context:
            raise HTTPException(status_code=404, detail="User context not found")

        return UserContextResponse(
            context_id=updated_context.context_id,
            user_id=updated_context.user_id,
            course_id=updated_context.course_id,
            user_context=updated_context.user_context,
            created_at=updated_context.created_at.isoformat(),
            updated_at=updated_context.updated_at.isoformat(),
            last_used_at=updated_context.last_used_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to update user context {context_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user context")

@router.delete("/{context_id}")
async def delete_user_context(context_id: str):
    """Delete user context"""
    try:
        success = await AsyncCRUD.delete(context_id)
        if not success:
            raise HTTPException(status_code=404, detail="User context not found")

        return {"message": "User context deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to delete user context {context_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user context")

@router.get("/stats/overview")
async def get_user_context_stats():
    """Get user context statistics"""
    try:
        async with async_db.get_session() as db:
            total_contexts = await db.execute(select(UserContext))
            total_count = len(total_contexts.scalars().all())

            template_count = 0
            # Count template-based contexts (those with template_id prefix)
            all_contexts = await AsyncCRUD.get_user_contexts(limit=10000)
            template_count = sum(1 for ctx in all_contexts if "template_id:" in ctx.user_context)

            return {
                "total_contexts": total_count,
                "template_based_contexts": template_count,
                "custom_contexts": total_count - template_count,
                "available_templates": len(DEFAULT_TEMPLATES)
            }
    except Exception as e:
        api_logger.error(f"Failed to get user context stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")