"""
User Management API
CRUD operations for user management with proper database integration
"""
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ..core.database import async_db, AsyncCRUD
from ..schemas.db_models import User
from ..core.logger import api_logger

router = APIRouter(prefix="/users", tags=["users"])

# Pydantic models for request/response
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., max_length=255, description="Email address")
    password_hash: str = Field(..., min_length=60, max_length=60, description="Password hash")
    role: str = Field(..., pattern="^(admin|instructor|student)$", description="User role")

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    created_at: str

class UserListResponse(BaseModel):
    users: List[UserResponse]
    total_count: int
    page: int
    limit: int

# CRUD operations using AsyncCRUD base class
class UserCRUD(AsyncCRUD):
    def __init__(self):
        super().__init__(User)

    async def get_users_by_role(
        self,
        role: str,
        page: int = 1,
        limit: int = 20
    ) -> tuple[List[User], int]:
        """Get paginated users by role."""
        async with async_db.get_session() as session:
            # Get total count
            count_result = await session.execute(
                select(func.count(User.user_id)).where(User.role == role)
            )
            total_count = count_result.scalar()

            # Get paginated results
            result = await session.execute(
                select(User)
                .where(User.role == role)
                .order_by(User.username)
                .limit(limit)
                .offset((page - 1) * limit)
            )

            users = result.scalars().all()
            return list(users), total_count

    async def search_users(
        self,
        query: str,
        page: int = 1,
        limit: int = 20
    ) -> tuple[List[User], int]:
        """Search users by username or email."""
        async with async_db.get_session() as session:
            search_pattern = f"%{query}%"

            # Get total count
            count_result = await session.execute(
                select(func.count(User.user_id))
                .where(
                    or_(
                        User.username.ilike(search_pattern),
                        User.email.ilike(search_pattern)
                    )
                )
            )
            total_count = count_result.scalar()

            # Get paginated results
            result = await session.execute(
                select(User)
                .where(
                    or_(
                        User.username.ilike(search_pattern),
                        User.email.ilike(search_pattern)
                    )
                )
                .order_by(User.username)
                .limit(limit)
                .offset((page - 1) * limit)
            )

            users = result.scalars().all()
            return list(users), total_count

# Initialize CRUD
user_crud = UserCRUD()

@router.get("/", response_model=UserListResponse)
async def list_users(
    role: Optional[str] = Query(None, description="Filter by role (admin, instructor, student)"),
    search: Optional[str] = Query(None, description="Search users by username or email"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
) -> UserListResponse:
    """List users with pagination and filtering."""
    try:
        if role and search:
            raise HTTPException(
                status_code=400,
                detail="Cannot use both role and search parameters together"
            )

        if role:
            users, total_count = await user_crud.get_users_by_role(
                role=role,
                page=page,
                limit=limit
            )
        elif search:
            users, total_count = await user_crud.search_users(
                query=search,
                page=page,
                limit=limit
            )
        else:
            # Get all users with pagination
            async with async_db.get_session() as session:
                # Get total count
                count_result = await session.execute(
                    select(func.count(User.user_id))
                )
                total_count = count_result.scalar()

                # Get paginated results
                result = await session.execute(
                    select(User)
                    .order_by(User.username)
                    .limit(limit)
                    .offset((page - 1) * limit)
                )

                users = result.scalars().all()
                users = list(users)

        user_responses = []
        for user in users:
            user_responses.append(UserResponse(
                user_id=user.user_id,
                username=user.username,
                email=user.email,
                role=user.role,
                created_at=user.created_at.isoformat()
            ))

        return UserListResponse(
            users=user_responses,
            total_count=total_count,
            page=page,
            limit=limit
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to list users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str) -> UserResponse:
    """Get user by ID."""
    try:
        async with async_db.get_session() as session:
            result = await session.execute(
                select(User).where(User.user_id == user_id)
            )
            user = result.scalar_one_or_none()

            if not user:
                raise HTTPException(status_code=404, detail=f"User {user_id} not found")

            return UserResponse(
                user_id=user.user_id,
                username=user.username,
                email=user.email,
                role=user.role,
                created_at=user.created_at.isoformat()
            )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=UserResponse, status_code=201)
async def create_user(user_data: UserCreate) -> UserResponse:
    """Create a new user."""
    try:
        api_logger.info(f"Creating user: {user_data.username}")

        async with async_db.get_session() as session:
            # Check if username already exists
            existing_user = await session.execute(
                select(User).where(User.username == user_data.username)
            ).scalar_one_or_none()

            if existing_user:
                raise HTTPException(
                    status_code=409,
                    detail=f"Username '{user_data.username}' already exists"
                )

            # Check if email already exists
            existing_email = await session.execute(
                select(User).where(User.email == user_data.email)
            ).scalar_one_or_none()

            if existing_email:
                raise HTTPException(
                    status_code=409,
                    detail=f"Email '{user_data.email}' already exists"
                )

        # Create user
        user_dict = user_data.model_dump()
        user = await user_crud.create(user_dict)

        api_logger.info(f"✅ Created user {user.user_id}")

        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            role=user.role,
            created_at=user.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=500, detail=str(e))