# File: db_models.py
"""
SQLAlchemy models for LMS Summary database + AI-Tutor Chatbot
UPDATED: Fixed to match actual database schema + AI-Tutor integration
"""
import uuid
import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, DateTime, Enum, Index, JSON, Boolean, text
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database import Base

def generate_uuid():
    """Generate a UUID string."""
    return str(uuid.uuid4())

class User(Base):
    """User model."""
    __tablename__ = "users"
    
    user_id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum("admin", "instructor", "student", name="user_role"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    courses = relationship("Course", back_populates="instructor")
    request_tracking = relationship("RequestTracking", back_populates="user")
    question_generations = relationship("QuestionGeneration", back_populates="user")

    # AI-Tutor relationships
    chatrooms = relationship("Chatroom", back_populates="user")
    messages = relationship("Message", back_populates="user")
    responses = relationship("Response", back_populates="user")
    user_contexts = relationship("UserContext", back_populates="user")

class Course(Base):
    """Course model."""
    __tablename__ = "courses"
    
    course_id = Column(String(36), primary_key=True, default=generate_uuid)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    instructor_id = Column(String(36), ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    instructor = relationship("User", back_populates="courses")
    documents = relationship("Document", back_populates="course", cascade="all, delete-orphan")
    question_generations = relationship("QuestionGeneration", back_populates="course")
    request_tracking = relationship("RequestTracking", back_populates="course")

    # AI-Tutor relationships
    chatrooms = relationship("Chatroom", back_populates="course")
    user_contexts = relationship("UserContext", back_populates="course")
    knowledge_base_materials = relationship("CourseKnowledgeBase", back_populates="course")
    
    @classmethod
    async def aget_or_create(cls, db, title: str, instructor_id: str = "default_user") -> "Course":
        """Async version of get_or_create."""
        from sqlalchemy import select

        try:
            result = await db.execute(select(cls).filter_by(title=title))
            course = result.scalar_one_or_none()

            if not course:
                course = cls(title=title, instructor_id=instructor_id)
                db.add(course)
                await db.flush()
                await db.refresh(course)

            return course
        except Exception as e:
            # Fallback: create new instance
            return cls(title=title, instructor_id=instructor_id)

    @classmethod
    def get_or_create(cls, db: Session, title: str, instructor_id: str = "default_user") -> "Course":
        """Get existing course or create new one."""
        course = db.query(cls).filter_by(title=title).first()
        if not course:
            course = cls(title=title, instructor_id=instructor_id)
            db.add(course)
            db.commit()
            db.refresh(course)
        return course

class Document(Base):
    """Document model for file uploads and processing tracking."""
    __tablename__ = "documents"

    document_id = Column(String(36), primary_key=True, default=generate_uuid)
    course_id = Column(String(36), ForeignKey("courses.course_id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow, server_default=text("now()"))
    md5_hash = Column(String(32), nullable=False, unique=True)
    processing_status = Column(String(20), default="pending", server_default=text("'pending'"))
    has_embeddings = Column(Boolean, default=False, server_default=text("false"))
    embedding_model = Column(String(100), nullable=True)

    # Relationships
    course = relationship("Course", back_populates="documents")
    request_tracking = relationship("RequestTracking", back_populates="document")

    # Indexes
    __table_args__ = (
        Index("idx_document_md5", "md5_hash"),
        Index("idx_document_status", "processing_status"),
        Index("idx_document_course", "course_id"),
    )

    @classmethod
    def create_from_upload(
        cls,
        db: Session,
        course_id: str,
        file_name: str,
        material_id: str,
        file_size: int,
        file_type: str = "pdf"
    ) -> "Document":
        """Create document record from file upload."""
        doc = cls(
            course_id=course_id,
            title=file_name,
            file_path=str(file_name),
            file_type=file_type,
            file_size=file_size,
            md5_hash=material_id,
            processing_status="pending"
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc

    def update_status(self, db: Session, status: str, has_embeddings: bool = False, embedding_model: str = None):
        """Update document processing status."""
        self.processing_status = status
        if has_embeddings:
            self.has_embeddings = has_embeddings
        if embedding_model:
            self.embedding_model = embedding_model
        db.commit()

class Content(Base):
    """Simplified content model for storing educational material."""
    __tablename__ = "contents"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid, server_default=text("gen_random_uuid()"))
    title = Column(String(500), nullable=False)
    content_text = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=True)
    processing_status = Column(String(50), nullable=True)
    source = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow, server_default=text("now()"))
    updated_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    questions = relationship("GeneratedQuestion", back_populates="content", cascade="all, delete-orphan")

    @classmethod
    def create_content(cls, db: Session, title: str, content_text: str, **kwargs) -> "Content":
        """Create new content record."""
        content = cls(
            title=title,
            content_text=content_text,
            **kwargs
        )
        db.add(content)
        db.commit()
        db.refresh(content)
        return content


class Summary(Base):
    """Summary model matching actual database schema with JSON content."""
    __tablename__ = "summaries"

    summary_id = Column(String(36), primary_key=True, default=generate_uuid)
    course_id = Column(String(36), nullable=True)
    content_id = Column(String(36), nullable=True)
    summary_type = Column(Enum("document", "course", "custom", name="summary_type"), nullable=False)
    content = Column(JSON, nullable=False)
    clustering_method = Column(String(50), nullable=True)
    num_clusters = Column(Integer, nullable=True)
    chunk_size = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    @property
    def material_id(self):
        """Get file hash from associated document or generate from content."""
        if self.content and isinstance(self.content, dict):
            metadata = self.content.get("metadata", {})
            material_ids = metadata.get("material_ids", [])
            if material_ids:
                return material_ids[0]

        # Generate hash from content as fallback
        import hashlib
        content_str = str(self.content) if self.content else ""
        return hashlib.md5(content_str.encode()).hexdigest()

    @property
    def file_name(self):
        """Get file name from content metadata or generate default."""
        if self.content and isinstance(self.content, dict):
            metadata = self.content.get("metadata", {})
            file_name = metadata.get("file_name")
            if file_name:
                return file_name

        return f"{self.summary_type}_summary_{self.summary_id[:8]}"

    @property
    def expires_at(self):
        """Calculate expiration based on created_at and default TTL."""
        import datetime
        default_ttl_hours = 168  # 7 days

        # Check if content contains custom TTL
        if self.content and isinstance(self.content, dict):
            metadata = self.content.get("metadata", {})
            custom_ttl = metadata.get("cache_ttl_hours")
            if custom_ttl:
                default_ttl_hours = min(custom_ttl, 720)  # Max 30 days

        return self.created_at + datetime.timedelta(hours=default_ttl_hours)

    @property
    def ttl_hours(self):
        """Get TTL hours from content metadata or default."""
        if self.content and isinstance(self.content, dict):
            metadata = self.content.get("metadata", {})
            custom_ttl = metadata.get("cache_ttl_hours")
            if custom_ttl:
                return min(custom_ttl, 720)  # Max 30 days

        return 168  # Default 7 days

    @property
    def cache_key(self):
        """Generate cache key."""
        return f"summary:{self.summary_type}:{self.material_id}"
    
    @classmethod
    def save_document_summary(
        cls,
        db: Session,
        document_id: str,
        summary_content: dict,
        method: str,
        num_clusters: int = None,
        material_id: str = None,
        file_name: str = None,
        cache_key: str = None,
        ttl_hours: int = 168
    ) -> "Summary":
        """Save document summary to database using existing structure."""
        # Add TTL and metadata to content
        if not isinstance(summary_content, dict):
            summary_content = {"result": summary_content}

        if "metadata" not in summary_content:
            summary_content["metadata"] = {}

        # Store custom TTL in metadata
        if ttl_hours and ttl_hours != 168:
            summary_content["metadata"]["cache_ttl_hours"] = min(ttl_hours, 720)

        if file_name:
            summary_content["metadata"]["file_name"] = file_name

        summary = cls(
            content_id=document_id,
            summary_type="document",
            content=summary_content,
            clustering_method=method,
            num_clusters=num_clusters,
            chunk_size=2000
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)
        return summary
    
    @classmethod
    def save_course_summary(
        cls,
        db: Session,
        course_id: str,
        summary_content: dict,
        method: str,
        num_clusters: int = None,
        material_ids: list = None,
        ttl_hours: int = 168
    ) -> "Summary":
        """Save course summary to database using existing structure."""
        # Add TTL and metadata to content
        if not isinstance(summary_content, dict):
            summary_content = {"result": summary_content}

        if "metadata" not in summary_content:
            summary_content["metadata"] = {}

        # Store file hashes and TTL in metadata
        if material_ids:
            summary_content["metadata"]["material_ids"] = material_ids

        if ttl_hours and ttl_hours != 168:
            summary_content["metadata"]["cache_ttl_hours"] = min(ttl_hours, 720)

        summary = cls(
            course_id=course_id,
            summary_type="course",
            content=summary_content,
            clustering_method=method,
            num_clusters=num_clusters
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)
        return summary

    @classmethod
    def find_existing_summary(
        cls,
        db: Session,
        material_id: str,
        summary_type: str = "document"
    ) -> "Summary":
        """Find existing valid summary by file hash using existing structure."""
        from datetime import datetime

        # Look for summaries with matching material_id in metadata
        summaries = db.query(cls).filter(
            cls.summary_type == summary_type
        ).order_by(cls.created_at.desc()).all()

        # Filter by material_id and expiration
        for summary in summaries:
            if not summary.is_expired():
                if summary.content and isinstance(summary.content, dict):
                    metadata = summary.content.get("metadata", {})
                    stored_ids = metadata.get("material_ids", [])
                    if material_id in stored_ids or summary.material_id == material_id:
                        return summary

        return None

    @classmethod
    def find_course_summary(
        cls,
        db: Session,
        material_ids: list,
        summary_type: str = "course"
    ) -> "Summary":
        """Find existing course summary by file hashes using existing structure."""
        from datetime import datetime

        # Look for course summaries and check metadata
        summaries = db.query(cls).filter(
            cls.summary_type == summary_type
        ).order_by(cls.created_at.desc()).all()

        # Find summary with matching file hashes in metadata
        for summary in summaries:
            if not summary.is_expired():
                if summary.content and isinstance(summary.content, dict):
                    metadata = summary.content.get("metadata", {})
                    stored_hashes = metadata.get("material_ids", [])

                    if sorted(stored_hashes) == sorted(material_ids):
                        return summary

        return None

    def is_expired(self) -> bool:
        """Check if summary has expired based on content metadata."""
        import datetime

        if not self.content or not isinstance(self.content, dict):
            expiration_date = self.created_at.replace(tzinfo=None) + datetime.timedelta(hours=168)
            return datetime.datetime.now(datetime.UTC) > expiration_date

        metadata = self.content.get("metadata", {})
        custom_ttl = metadata.get("cache_ttl_hours", 168)
        custom_ttl = min(custom_ttl, 720)

        expiration_date = self.created_at.replace(tzinfo=None) + datetime.timedelta(hours=custom_ttl)
        return datetime.datetime.now(datetime.UTC) > expiration_date

    def extend_ttl(self, db: Session, additional_hours: int = 168):
        """Extend TTL for this summary by updating metadata."""
        import datetime

        if not self.content or not isinstance(self.content, dict):
            self.content = {"metadata": {}}
        elif "metadata" not in self.content:
            self.content["metadata"] = {}

        self.content["metadata"]["cache_ttl_hours"] = min(additional_hours, 720)
        self.content["metadata"]["extended_at"] = datetime.datetime.now(datetime.UTC).isoformat()

        db.commit()
        db.refresh(self)

class RequestTracking(Base):
    """General request tracking model for all service requests."""
    __tablename__ = "request_tracking"

    tracking_id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True)
    request_type = Column(String(20), nullable=False)
    service = Column(String(50), nullable=False)
    endpoint = Column(String(255), nullable=False)

    # Source tracking
    document_id = Column(String(36), ForeignKey("documents.document_id", ondelete="SET NULL"), nullable=True)
    course_id = Column(String(36), ForeignKey("courses.course_id", ondelete="SET NULL"), nullable=True)
    file_hashes = Column(JSON, nullable=True)

    # Request parameters
    parameters = Column(JSON)

    # Status tracking
    status = Column(String(20), default="pending", server_default=text("'pending'"))
    created_at = Column(DateTime, default=datetime.datetime.utcnow, server_default=text("now()"))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Results tracking
    tokens_used = Column(Integer, default=0, server_default=text("0"))
    result_count = Column(Integer, default=0, server_default=text("0"))
    error_message = Column(Text)
    request_metadata = Column(JSON)

    # Relationships
    user = relationship("User", back_populates="request_tracking")
    document = relationship("Document", back_populates="request_tracking")
    course = relationship("Course", back_populates="request_tracking")

    # Indexes
    __table_args__ = (
        Index("idx_request_tracking_user_id", "user_id"),
        Index("idx_request_tracking_status", "status"),
        Index("idx_request_tracking_type_service", "request_type", "service"),
        Index("idx_request_tracking_created_at", "created_at"),
    )

    @classmethod
    async def acreate_request(
        cls,
        db,
        request_type: str,
        service: str,
        endpoint: str,
        user_id: str = None,
        document_id: str = None,
        course_id: str = None,
        material_ids: list = None,
        parameters: dict = None
    ) -> "RequestTracking":
        """Async version of create_request."""
        tracking = cls(
            user_id=user_id,
            request_type=request_type,
            service=service,
            endpoint=endpoint,
            document_id=document_id,
            course_id=course_id,
            file_hashes=material_ids or [],
            parameters=parameters or {},
            created_at=datetime.datetime.utcnow()
        )
        db.add(tracking)
        await db.flush()
        await db.refresh(tracking)
        return tracking

    @classmethod
    def create_request(
        cls,
        db: Session,
        request_type: str,
        service: str,
        endpoint: str,
        user_id: str = None,
        document_id: str = None,
        course_id: str = None,
        material_ids: list = None,
        parameters: dict = None
    ) -> "RequestTracking":
        """Create new request tracking record."""
        tracking = cls(
            user_id=user_id,
            request_type=request_type,
            service=service,
            endpoint=endpoint,
            document_id=document_id,
            course_id=course_id,
            file_hashes=material_ids or [],
            parameters=parameters or {},
            created_at=datetime.datetime.utcnow()
        )
        db.add(tracking)
        db.commit()
        db.refresh(tracking)
        return tracking

    def start_processing(self, db: Session):
        """Mark request as processing."""
        self.status = "processing"
        self.started_at = datetime.datetime.utcnow()
        db.commit()

    def complete(self, db: Session, result_count: int = 0, tokens_used: int = 0, request_metadata: dict = None):
        """Mark request as completed."""
        self.status = "completed"
        self.completed_at = datetime.datetime.utcnow()
        self.result_count = result_count
        self.tokens_used = tokens_used
        if request_metadata:
            self.request_metadata = request_metadata
        db.commit()

    def fail(self, db: Session, error_message: str):
        """Mark request as failed."""
        self.status = "failed"
        self.completed_at = datetime.datetime.utcnow()
        self.error_message = error_message
        db.commit()

    @property
    def duration_ms(self) -> float:
        """Calculate request duration in milliseconds."""
        if self.completed_at and self.started_at:
            diff = self.completed_at - self.started_at
            return diff.total_seconds() * 1000
        return 0

class ApiLog(Base):
    """Simplified API log model."""
    __tablename__ = "api_logs"

    log_id = Column(String(36), primary_key=True, default=generate_uuid)
    request_id = Column(String(36), nullable=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    request_params = Column(JSON)
    response_code = Column(Integer)
    response_time = Column(Float)
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=True)
    ip_address = Column(String(45))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    user = relationship("User")
    
    @classmethod
    def log_api_call(
        cls,
        db: Session,
        endpoint: str,
        method: str,
        response_code: int,
        response_time: float,
        user_id: str = None,
        request_params: dict = None
    ):
        """Log API call."""
        log = cls(
            endpoint=endpoint,
            method=method,
            request_params=request_params,
            response_code=response_code,
            response_time=response_time,
            user_id=user_id
        )
        db.add(log)
        db.commit()


class QuestionGeneration(Base):
    """Question generation session model."""
    __tablename__ = "question_generations"

    generation_id = Column(String(36), primary_key=True, default=generate_uuid)
    course_id = Column(String(36), ForeignKey("courses.course_id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False)
    generation_type = Column(Enum("course", "document", name="generation_type"), nullable=False)
    parameters = Column(JSON)
    status = Column(Enum("pending", "processing", "completed", "failed", name="generation_status"), default="pending")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    tokens_used = Column(Integer)

    # Relationships
    course = relationship("Course", back_populates="question_generations")
    user = relationship("User", back_populates="question_generations")

    @classmethod
    def create_generation(
        cls,
        db: Session,
        course_id: str,
        user_id: str,
        generation_type: str = "course",
        parameters: dict = None
    ) -> "QuestionGeneration":
        """Create new question generation session."""
        generation = cls(
            course_id=course_id,
            user_id=user_id,
            generation_type=generation_type,
            parameters=parameters or {},
            status="processing"
        )
        db.add(generation)
        db.commit()
        db.refresh(generation)
        return generation

    def complete(self, db: Session, tokens_used: int):
        """Mark generation as completed."""
        self.status = "completed"
        self.completed_at = datetime.datetime.utcnow()
        self.tokens_used = tokens_used
        db.commit()

    def fail(self, db: Session, error_message: str):
        """Mark generation as failed."""
        self.status = "failed"
        self.completed_at = datetime.datetime.utcnow()
        self.error_message = error_message
        db.commit()


class AnswerKey(Base):
    """Answer key model for storing question answers."""
    __tablename__ = "answer_keys"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid, server_default=text("gen_random_uuid()"))
    question_id = Column(UUID(as_uuid=False), ForeignKey("questions.id", ondelete="CASCADE"), nullable=False)
    answer_text = Column(String(500), nullable=False)
    answer_type = Column(String(20), default="text", server_default=text("'text'"))
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow, server_default=text("now()"))

    # Relationships
    question = relationship("GeneratedQuestion", back_populates="answer_key")

    # Indexes
    __table_args__ = (
        Index("idx_answer_keys_question_id", "question_id"),
    )

    @classmethod
    def create_answer_key(cls, db: Session, question_id: str, answer_text: str, answer_type: str = "text", explanation: str = None) -> "AnswerKey":
        """Create new answer key."""
        answer_key = cls(
            question_id=question_id,
            answer_text=answer_text,
            answer_type=answer_type,
            explanation=explanation
        )
        db.add(answer_key)
        db.commit()
        db.refresh(answer_key)
        return answer_key


class QuestionOption(Base):
    """Question options model for MCQ answers."""
    __tablename__ = "question_options"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid, server_default=text("gen_random_uuid()"))
    question_id = Column(UUID(as_uuid=False), ForeignKey("questions.id", ondelete="CASCADE"), nullable=False)
    option_index = Column(Integer, nullable=False)
    option_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow, server_default=text("now()"))

    # Relationships
    question = relationship("GeneratedQuestion", back_populates="options")

    # Indexes
    __table_args__ = (
        Index("idx_question_options_question_id", "question_id"),
        Index("uk_question_option", "question_id", "option_index", unique=True),
    )

    @classmethod
    def create_options(cls, db: Session, question_id: str, options_list: list) -> list["QuestionOption"]:
        """Create multiple options for a question."""
        created_options = []
        for i, option_text in enumerate(options_list):
            option = cls(
                question_id=question_id,
                option_index=i,
                option_text=option_text
            )
            db.add(option)
            created_options.append(option)

        db.commit()
        for option in created_options:
            db.refresh(option)

        return created_options


class GeneratedQuestion(Base):
    """Individual generated question model."""
    __tablename__ = "questions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=generate_uuid, server_default=text("gen_random_uuid()"))
    content_id = Column(UUID(as_uuid=False), ForeignKey("contents.id", ondelete="CASCADE"), nullable=False)
    question_text = Column(Text, nullable=False)
    question_type = Column(String(50), nullable=False)
    difficulty = Column(String(20), nullable=True)
    cognitive_level = Column(Enum("C1", "C2", "C3", "C4", "C5", "C6", name="cognitive_level_enum"), default="C2", server_default=text("'C2'"))
    question_pattern = Column(Enum("P1", "P2", "P3", "P4", "P5", "P6", "P7", name="question_pattern_enum"), default="P1", server_default=text("'P1'"))
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow, server_default=text("now()"))
    updated_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow, server_default=text("now()"))

    # Relationships
    content = relationship("Content", back_populates="questions")
    answer_key = relationship("AnswerKey", back_populates="question", uselist=False)
    options = relationship("QuestionOption", back_populates="question")

    # Indexes
    __table_args__ = (
        Index("idx_questions_cognitive_level", "cognitive_level"),
        Index("idx_questions_question_pattern", "question_pattern"),
        Index("idx_questions_question_type", "question_type"),
        Index("idx_questions_difficulty", "difficulty"),
        Index("idx_questions_content_id", "content_id"),
    )

    @classmethod
    def create_question(
        cls,
        db: Session,
        content_id: str,
        question_data: dict
    ) -> "GeneratedQuestion":
        """Create new generated question with separate answer and options."""
        try:
            question = cls(
                content_id=content_id,
                question_text=question_data["question_text"],
                question_type=question_data["question_type"],
                difficulty=question_data.get("difficulty", "D2"),
                cognitive_level=question_data.get("cognitive_level", "C2"),
                question_pattern=question_data.get("question_pattern", "P1")
            )
            db.add(question)
            db.commit()
            db.refresh(question)

            # Create answer key separately
            answer_text = question_data.get("answer_text", "")
            if answer_text:
                answer_type = "text"
                if question_data.get("question_type") == "multiple_choice":
                    answer_type = "option_index"
                elif question_data.get("question_type") == "true_false":
                    answer_type = "boolean"

                AnswerKey.create_answer_key(
                    db,
                    question.id,
                    answer_text=answer_text,
                    answer_type=answer_type,
                    explanation=question_data.get("explanation")
                )

            # Create question options separately for MCQ
            options_list = question_data.get("options", [])
            if options_list and question_data["question_type"] == "multiple_choice":
                QuestionOption.create_options(db, question.id, options_list)

            return question

        except Exception as db_error:
            print(f"Error creating question: {db_error}")
            db.rollback()
            raise db_error

    @property
    def formatted_options(self) -> list:
        """Get formatted options for MCQ."""
        if self.question_type == "multiple_choice" and self.options:
            return self.options
        return []

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        answer_text = self.answer_key.answer_text if self.answer_key else ""
        
        return {
            "id": str(self.id),
            "content_id": str(self.content_id),
            "question_text": self.question_text,
            "question_type": self.question_type,
            "answer_text": answer_text,
            "options": [{"index": opt.option_index, "text": opt.option_text} for opt in self.formatted_options],
            "difficulty": self.difficulty,
            "cognitive_level": self.cognitive_level,
            "question_pattern": self.question_pattern,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


# ================================
# AI-TUTOR CHATBOT MODELS
# ================================

class Chatroom(Base):
    """Chatroom model for course-based AI-Tutor conversations."""
    __tablename__ = "chatrooms"

    chatroom_id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    course_id = Column(String(36), ForeignKey("courses.course_id", ondelete="CASCADE"), nullable=False)
    room_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Room settings
    is_active = Column(Boolean, default=True, server_default=text("true"))
    max_messages = Column(Integer, default=1000, server_default=text("1000"))

    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow, server_default=text("now()"))
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="chatrooms")
    course = relationship("Course", back_populates="chatrooms")
    messages = relationship("Message", back_populates="chatroom", cascade="all, delete-orphan")
    responses = relationship("Response", back_populates="chatroom", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_chatrooms_user_id", "user_id"),
        Index("idx_chatrooms_course_id", "course_id"),
        Index("idx_chatrooms_active", "is_active"),
        Index("uk_user_course_room", "user_id", "course_id", "room_name", unique=True),
    )

    @classmethod
    def create_chatroom(
        cls,
        db: Session,
        user_id: str,
        course_id: str,
        room_name: str,
        description: str = None
    ) -> "Chatroom":
        """Create new chatroom for user in specific course."""
        chatroom = cls(
            user_id=user_id,
            course_id=course_id,
            room_name=room_name,
            description=description
        )
        db.add(chatroom)
        db.commit()
        db.refresh(chatroom)
        return chatroom

    def update_activity(self, db: Session):
        """Update last activity timestamp."""
        self.last_activity_at = datetime.datetime.now(datetime.UTC)
        self.updated_at = datetime.datetime.now(datetime.UTC)
        db.commit()

    def to_dict(self) -> Dict[str, Any]:
        """Convert chatroom to dictionary."""
        return {
            "chatroom_id": self.chatroom_id,
            "user_id": self.user_id,
            "course_id": self.course_id,
            "room_name": self.room_name,
            "description": self.description,
            "is_active": self.is_active,
            "max_messages": self.max_messages,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat()
        }

    async def get_conversation_history(self, db: Session, limit: int = 5) -> str:
        """Get last N message-response pairs for ChatPromptTemplate."""
        from sqlalchemy import select, desc

        # Get last N messages with their responses
        result = await db.execute(
            select(Message, Response)
            .join(Response, Message.message_id == Response.message_id, isouter=True)
            .where(Message.chatroom_id == self.chatroom_id)
            .order_by(desc(Message.created_at))
            .limit(limit)
        )

        history_parts = []
        for row in result.fetchall():
            message, response = row
            history_parts.append(f"User: {message.message_text}")
            if response:
                history_parts.append(f"Assistant: {response.response_text}")

        return "\n".join(reversed(history_parts))  # Return in chronological order


class Message(Base):
    """Message model for storing user queries."""
    __tablename__ = "messages"

    message_id = Column(String(36), primary_key=True, default=generate_uuid)
    chatroom_id = Column(String(36), ForeignKey("chatrooms.chatroom_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)

    # Message content
    message_text = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow, server_default=text("now()"))

    # Relationships
    chatroom = relationship("Chatroom", back_populates="messages")
    user = relationship("User", back_populates="messages")
    responses = relationship("Response", back_populates="message", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index("idx_messages_chatroom_id", "chatroom_id"),
        Index("idx_messages_user_id", "user_id"),
        Index("idx_messages_created_at", "created_at"),
    )

    @classmethod
    async def acreate_message(
        cls,
        db,
        chatroom_id: str,
        user_id: str,
        message_text: str
    ) -> "Message":
        """Async version of create_message."""
        message = cls(
            chatroom_id=chatroom_id,
            user_id=user_id,
            message_text=message_text
        )
        db.add(message)
        await db.flush()
        await db.refresh(message)
        return message

    @classmethod
    def create_message(
        cls,
        db: Session,
        chatroom_id: str,
        user_id: str,
        message_text: str
    ) -> "Message":
        """Create new message in chatroom."""
        message = cls(
            chatroom_id=chatroom_id,
            user_id=user_id,
            message_text=message_text
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "chatroom_id": self.chatroom_id,
            "user_id": self.user_id,
            "message_text": self.message_text,
            "created_at": self.created_at.isoformat()
        }


class Response(Base):
    """AI-Tutor response model with comprehensive tracking."""
    __tablename__ = "responses"

    response_id = Column(String(36), primary_key=True, default=generate_uuid)
    message_id = Column(String(36), ForeignKey("messages.message_id", ondelete="CASCADE"), nullable=False)
    chatroom_id = Column(String(36), ForeignKey("chatrooms.chatroom_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)

    # Response content
    response_text = Column(Text, nullable=False)
    model_used = Column(String(100), nullable=False)  # gpt-4o, gpt-4o-mini, etc.

    # Response type and source
    response_type = Column(
        Enum("cache_hit_raw", "cache_hit_personalized", "rag_response", "error", name="response_type"),
        nullable=False
    )
    source_type = Column(Enum("redis_cache", "knowledge_base", name="source_type"), nullable=False)

    # Cache information
    cache_hit = Column(Boolean, default=False, server_default=text("false"))
    cache_similarity_score = Column(Float, nullable=True)  # Vector similarity score for cache hits

    # Personalization information
    personalized = Column(Boolean, default=False, server_default=text("false"))

    # Token usage and cost tracking
    input_tokens = Column(Integer, default=0, server_default=text("0"))
    output_tokens = Column(Integer, default=0, server_default=text("0"))
    total_tokens = Column(Integer, default=0, server_default=text("0"))
    cost_usd = Column(Float, default=0.0, server_default=text("0.0"))

    # Performance metrics
    latency_ms = Column(Float, nullable=False)  # Total response latency

    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow, server_default=text("now()"))
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    message = relationship("Message", back_populates="responses")
    chatroom = relationship("Chatroom", back_populates="responses")
    user = relationship("User", back_populates="responses")

    # Indexes
    __table_args__ = (
        Index("idx_responses_message_id", "message_id"),
        Index("idx_responses_chatroom_id", "chatroom_id"),
        Index("idx_responses_user_id", "user_id"),
        Index("idx_responses_type", "response_type"),
        Index("idx_responses_cache_hit", "cache_hit"),
        Index("idx_responses_created_at", "created_at"),
        Index("idx_responses_model_used", "model_used"),
    )

    @classmethod
    async def acreate_response(
        cls,
        db,
        message_id: str,
        chatroom_id: str,
        user_id: str,
        response_text: str,
        model_used: str,
        response_type: str,
        source_type: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        cache_hit: bool = False,
        cache_similarity_score: float = None,
        personalized: bool = False
    ) -> "Response":
        """Async version of create_response."""
        response = cls(
            message_id=message_id,
            chatroom_id=chatroom_id,
            user_id=user_id,
            response_text=response_text,
            model_used=model_used,
            response_type=response_type,
            source_type=source_type,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
            cache_hit=cache_hit,
            cache_similarity_score=cache_similarity_score,
            personalized=personalized
        )
        db.add(response)
        await db.flush()
        await db.refresh(response)
        return response

    @classmethod
    def create_response(
        cls,
        db: Session,
        message_id: str,
        chatroom_id: str,
        user_id: str,
        response_text: str,
        model_used: str,
        response_type: str,
        source_type: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        cache_hit: bool = False,
        cache_similarity_score: float = None,
        personalized: bool = False
    ) -> "Response":
        """Create new AI-Tutor response with essential tracking."""
        response = cls(
            message_id=message_id,
            chatroom_id=chatroom_id,
            user_id=user_id,
            response_text=response_text,
            model_used=model_used,
            response_type=response_type,
            source_type=source_type,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
            cache_hit=cache_hit,
            cache_similarity_score=cache_similarity_score,
            personalized=personalized
        )
        db.add(response)
        db.commit()
        db.refresh(response)
        return response

    def update_rating(self, db: Session, rating: int, feedback: str = None):
        """Update user rating and feedback."""
        self.user_rating = rating
        self.user_feedback = feedback
        self.updated_at = datetime.datetime.now(datetime.UTC)
        db.commit()

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to comprehensive dictionary."""
        return {
            "response_id": self.response_id,
            "message_id": self.message_id,
            "chatroom_id": self.chatroom_id,
            "user_id": self.user_id,
            "response_text": self.response_text,
            "model_used": self.model_used,
            "response_type": self.response_type,
            "source_type": self.source_type,
            "cache_hit": self.cache_hit,
            "cache_key": self.cache_key,
            "cache_similarity_score": self.cache_similarity_score,
            "personalized": self.personalized,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "user_rating": self.user_rating,
            "user_feedback": self.user_feedback,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class UserContext(Base):
    """User context model for personalization."""
    __tablename__ = "user_contexts"

    context_id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    course_id = Column(String(36), ForeignKey("courses.course_id", ondelete="CASCADE"), nullable=False)

    # User context - single field for easy ChatPromptTemplate integration
    user_context = Column(Text, nullable=True)  # User preferences, goals, COSTAR framework data

    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow, server_default=text("now()"))
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    last_used_at = Column(DateTime, default=datetime.datetime.utcnow, server_default=text("now()"))

    # Relationships
    user = relationship("User", back_populates="user_contexts")
    course = relationship("Course", back_populates="user_contexts")

    # Indexes
    __table_args__ = (
        Index("idx_user_contexts_user_id", "user_id"),
        Index("idx_user_contexts_course_id", "course_id"),
        Index("idx_user_contexts_last_used", "last_used_at"),
        Index("uk_user_course_context", "user_id", "course_id", unique=True),
    )

    @classmethod
    async def aget_or_create(
        cls,
        db,
        user_id: str,
        course_id: str,
        initial_context: str = None
    ) -> "UserContext":
        """Async version of get_or_create."""
        from sqlalchemy import select

        try:
            result = await db.execute(select(cls).filter_by(user_id=user_id, course_id=course_id))
            context = result.scalar_one_or_none()

            if not context:
                context = cls(
                    user_id=user_id,
                    course_id=course_id,
                    user_context=initial_context or ""
                )
                db.add(context)
                await db.flush()
                await db.refresh(context)

            return context
        except Exception as e:
            # Fallback: create new instance without database query
            return cls(
                user_id=user_id,
                course_id=course_id,
                user_context=initial_context or ""
            )

    @classmethod
    def get_or_create(
        cls,
        db: Session,
        user_id: str,
        course_id: str,
        initial_context: str = None
    ) -> "UserContext":
        """Sync version of get_or_create."""
        context = db.query(cls).filter_by(user_id=user_id, course_id=course_id).first()

        if not context:
            context = cls(
                user_id=user_id,
                course_id=course_id,
                user_context=initial_context or ""
            )
            db.add(context)
            db.commit()
            db.refresh(context)

        return context

    def update_context(self, db: Session, new_context: str):
        """Update user context."""
        self.user_context = new_context
        self.updated_at = datetime.datetime.now(datetime.UTC)
        db.commit()

    def get_context(self) -> str:
        """Get user context for ChatPromptTemplate."""
        return self.user_context or ""


class CourseKnowledgeBase(Base):
    """Link table between courses and knowledge base materials."""
    __tablename__ = "course_knowledge_bases"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    course_id = Column(String(36), ForeignKey("courses.course_id", ondelete="CASCADE"), nullable=False)
    material_id = Column(String(32), nullable=False)  # MD5 hash of document
    material_type = Column(Enum("document", "content", "external", name="material_type"), nullable=False)

    # Material metadata
    title = Column(String(500), nullable=False)
    file_name = Column(String(255), nullable=True)
    file_path = Column(String(1000), nullable=True)
    file_size = Column(Integer, nullable=True)
    chunk_count = Column(Integer, default=0, server_default=text("0"))

    # Processing status
    processed = Column(Boolean, default=False, server_default=text("false"))
    embedding_model = Column(String(100), nullable=True)

    # Usage statistics
    access_count = Column(Integer, default=0, server_default=text("0"))
    last_accessed_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow, server_default=text("now()"))
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    course = relationship("Course", back_populates="knowledge_base_materials")

    # Indexes
    __table_args__ = (
        Index("idx_course_kb_course_id", "course_id"),
        Index("idx_course_kb_material_id", "material_id"),
        Index("idx_course_kb_processed", "processed"),
        Index("idx_course_kb_type", "material_type"),
        Index("uk_course_material", "course_id", "material_id", unique=True),
    )

    @classmethod
    def add_material(
        cls,
        db: Session,
        course_id: str,
        material_id: str,
        material_type: str,
        title: str,
        file_name: str = None,
        file_path: str = None,
        file_size: int = None
    ) -> "CourseKnowledgeBase":
        """Add material to course knowledge base."""
        material = db.query(cls).filter_by(
            course_id=course_id,
            material_id=material_id
        ).first()

        if not material:
            material = cls(
                course_id=course_id,
                material_id=material_id,
                material_type=material_type,
                title=title,
                file_name=file_name,
                file_path=file_path,
                file_size=file_size
            )
            db.add(material)
            db.commit()
            db.refresh(material)

        return material

    def mark_processed(self, db: Session, embedding_model: str, chunk_count: int = 0):
        """Mark material as processed."""
        self.processed = True
        self.embedding_model = embedding_model
        self.chunk_count = chunk_count
        self.updated_at = datetime.datetime.now(datetime.UTC)
        db.commit()

    def record_access(self, db: Session):
        """Record material access."""
        self.access_count += 1
        self.last_accessed_at = datetime.datetime.now(datetime.UTC)
        db.commit()