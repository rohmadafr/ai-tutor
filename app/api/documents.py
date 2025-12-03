"""
Document Management API for RAG Knowledge Base
Handles document upload, ingestion, search, and deletion operations
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, Optional
import tempfile
import csv
import io
from pathlib import Path

from sqlalchemy import select, and_

from ..services.unified_rag_service import UnifiedRAGService
from ..services.custom_cache_service import CustomCacheService
from ..utils.pdf_extractor import PDFExtractor
from ..utils.file_hasher import FileHasher
from ..utils.text_preprocessing import text_preprocessor
from ..core.logger import api_logger
from ..core.database import async_db

router = APIRouter(prefix="/documents", tags=["documents"])

# Global service instances
_rag_service: Optional[UnifiedRAGService] = None
_cache_service: Optional[CustomCacheService] = None


async def get_rag_service() -> UnifiedRAGService:
    """Get RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = UnifiedRAGService()
    return _rag_service


async def get_cache_service() -> CustomCacheService:
    """Get cache service instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CustomCacheService()
        await _cache_service.connect()
    return _cache_service


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    course_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload and process a document (PDF, TXT) for RAG knowledge base
    All documents are automatically preprocessed for optimal quality.

    Args:
        file: Uploaded file (PDF, TXT)
        course_id: Optional course ID for organization

    Returns:
        Processing results and document IDs
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.txt'}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            api_logger.info(f"Processing uploaded file: {file.filename} ({file_extension})")

            # Generate material ID from file content
            material_id = FileHasher.get_material_id(Path(temp_file_path))
            api_logger.info(f"Generated material_id: {material_id}")

            # Check if material already exists in Redis (knowledge base)
            rag_service = await get_rag_service()
            existing_material_docs = await rag_service.get_documents_by_material_id(material_id, limit=10000)  # Get all chunks
            total_existing_chunks = len(existing_material_docs)

            if existing_material_docs:
                api_logger.info(f"📄 Material {material_id} already exists in knowledge base with {total_existing_chunks} chunks, skipping embedding generation")

                # Update database record for this course if needed
                if course_id:
                    async with async_db.get_session() as session:
                        from ..schemas.db_models import CourseKnowledgeBase

                        existing_kb = await session.execute(
                            select(CourseKnowledgeBase).where(
                                and_(
                                    CourseKnowledgeBase.course_id == course_id,
                                    CourseKnowledgeBase.material_id == material_id
                                )
                            )
                        )
                        kb_entry = existing_kb.scalar_one_or_none()

                        if not kb_entry:
                            # Create knowledge base entry for this course
                            kb_entry = CourseKnowledgeBase(
                                course_id=course_id,
                                material_id=material_id,
                                material_type="document",
                                file_path=file.filename,
                                file_size=len(content),
                                processed=True,
                                embedding_model=getattr(rag_service, 'embedding_model', 'text-embedding-3-small'),
                                chunk_count=total_existing_chunks,  # Use actual count from existing docs
                                access_count=0
                            )
                            session.add(kb_entry)
                            await session.commit()
                            await session.refresh(kb_entry)
                            api_logger.info(f"✅ Linked existing material {material_id} to course {course_id} with {total_existing_chunks} chunks")

                # Clean up temporary file
                Path(temp_file_path).unlink(missing_ok=True)

                return {
                    "message": f"Document {file.filename} already exists in knowledge base",
                    "filename": file.filename,
                    "material_id": material_id,
                    "course_id": course_id,
                    "total_chunks": total_existing_chunks,
                    "file_type": file_extension,
                    "status": "already_processed",
                    "action": "linked_to_course" if course_id else "already_exists"
                }

            # Use original filename for storage
            api_logger.info(f"Processing file with original filename: {file.filename}")

            documents = []

            if file_extension == '.pdf':
                # Process PDF with automatic preprocessing
                pdf_extractor = PDFExtractor(preprocess_text=True)
                langchain_docs = pdf_extractor.extract(Path(temp_file_path), file.filename)

                # Convert LangChain docs to our format
                for doc in langchain_docs:
                    documents.append({
                        "content": doc.page_content,
                        "metadata": {
                            "material_id": material_id,
                            "course_id": course_id or "default",
                            "filename": file.filename,
                            "file_type": "pdf",
                            "preprocessed": True,
                            **doc.metadata  # This includes page, source, and other PDF metadata
                        }
                    })

            elif file_extension == '.txt':
                # Process text file with automatic preprocessing
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()

                # Always apply preprocessing
                text_content = text_preprocessor.preprocess_text(text_content)

                documents.append({
                    "content": text_content,
                    "metadata": {
                        "material_id": material_id,
                        "course_id": course_id or "default",
                        "filename": file.filename,
                        "file_type": "txt",
                        "preprocessed": True
                    }
                })

            # Store in RAG knowledge base using existing method
            rag_service = await get_rag_service()
            await rag_service.add_documents(documents)

            # Create record in documents table
            async with async_db.get_session() as session:
                from ..schemas.db_models import Document

                # Check if document already exists
                existing_doc_result = await session.execute(
                    select(Document).where(Document.md5_hash == material_id)
                )
                existing_doc = existing_doc_result.scalar_one_or_none()

                if not existing_doc:
                    # Create new document record
                    doc_entry = Document(
                        title=file.filename,
                        file_path=file.filename,
                        file_type=file_extension,
                        file_size=len(content),
                        md5_hash=material_id,
                        processing_status="completed",
                        has_embeddings=True,
                        embedding_model=getattr(rag_service, 'embedding_model', 'text-embedding-3-small')
                    )

                    # Assign course_id if provided
                    if course_id:
                        doc_entry.course_id = course_id

                    session.add(doc_entry)
                    await session.commit()
                    await session.refresh(doc_entry)

                    api_logger.info(f"✅ Created document record for {file.filename}")
                else:
                    # Update existing document
                    existing_doc.processing_status = "completed"
                    existing_doc.has_embeddings = True
                    existing_doc.file_size = len(content)
                    existing_doc.embedding_model = getattr(rag_service, 'embedding_model', 'text-embedding-3-small')

                    if course_id:
                        existing_doc.course_id = course_id

                    await session.commit()
                    api_logger.info(f"📝 Updated document record for {file.filename}")

            # Create record in course_knowledge_bases table if course_id provided
            if course_id:
                async with async_db.get_session() as session:
                    from ..schemas.db_models import CourseKnowledgeBase

                    # Check if entry already exists
                    existing_result = await session.execute(
                        select(CourseKnowledgeBase).where(
                            and_(
                                CourseKnowledgeBase.course_id == course_id,
                                CourseKnowledgeBase.material_id == material_id
                            )
                        )
                    )
                    existing_kb = existing_result.scalar_one_or_none()

                    if not existing_kb:
                        # Create new knowledge base entry
                        kb_entry = CourseKnowledgeBase(
                            course_id=course_id,
                            material_id=material_id,
                            material_type="document",
                            title=file.filename,
                            file_name=file.filename,
                            file_path=file.filename,
                            file_size=len(content),
                            processed=True,
                            embedding_model=getattr(rag_service, 'embedding_model', 'text-embedding-3-small'),
                            chunk_count=len(documents),
                            access_count=0
                        )
                        session.add(kb_entry)
                        await session.commit()
                        await session.refresh(kb_entry)

                        api_logger.info(f"✅ Created knowledge base entry for {file.filename} in course {course_id}")
                    else:
                        # Update existing entry
                        existing_kb.chunk_count = len(documents)
                        existing_kb.processed = True
                        existing_kb.file_size = len(content)
                        existing_kb.embedding_model = getattr(rag_service, 'embedding_model', 'text-embedding-3-small')
                        await session.commit()

                        api_logger.info(f"📝 Updated knowledge base entry for {file.filename} in course {course_id}")

            api_logger.info(f"Successfully processed {file.filename}: {len(documents)} chunks, material_id: {material_id}")

            return {
                "message": f"Successfully uploaded and processed {file.filename}",
                "filename": file.filename,
                "material_id": material_id,
                "course_id": course_id,
                "total_chunks": len(documents),
                "file_type": file_extension,
                "preprocessed": True,
                "total_chars": sum(len(doc["content"]) for doc in documents),
                "knowledge_base_linked": course_id is not None,
                "status": "newly_processed",
                "action": "processed_and_stored"
            }

        finally:
            # Clean up temporary file
            Path(temp_file_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to upload document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get("/course/{course_id}")
async def get_course_documents(
    course_id: str,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get all documents for a specific course_id

    Args:
        course_id: Course ID to retrieve documents for
        limit: Maximum number of documents to return

    Returns:
        List of documents with metadata
    """
    try:
        if not course_id or not course_id.strip():
            raise HTTPException(status_code=400, detail="Course ID cannot be empty")

        if limit < 1 or limit > 5000:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 5000")

        rag_service = await get_rag_service()
        documents = await rag_service.get_documents_by_course_id(course_id, limit)

        if not documents:
            raise HTTPException(status_code=404, detail=f"No documents found for course_id: {course_id}")

        return {
            "course_id": course_id,
            "documents": documents,
            "total_documents": len(documents),
            "total_chars": sum(len(doc.get("text", "")) for doc in documents),
            "limit": limit
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get course documents {course_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@router.get("/material/{material_id}")
async def get_material_documents(
    material_id: str,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get all documents for a specific material_id

    Args:
        material_id: Material ID to retrieve documents for
        limit: Maximum number of documents to return

    Returns:
        List of documents with metadata
    """
    try:
        if not material_id or not material_id.strip():
            raise HTTPException(status_code=400, detail="Material ID cannot be empty")

        if limit < 1 or limit > 5000:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 5000")

        rag_service = await get_rag_service()
        documents = await rag_service.get_documents_by_material_id(material_id, limit)

        if not documents:
            raise HTTPException(status_code=404, detail=f"No documents found for material_id: {material_id}")

        return {
            "material_id": material_id,
            "documents": documents,
            "total_documents": len(documents),
            "total_chars": sum(len(doc.get("text", "")) for doc in documents),
            "limit": limit
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get material documents {material_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@router.delete("/course/{course_id}")
async def delete_course_documents(course_id: str) -> Dict[str, Any]:
    """
    Delete all documents for a specific course_id

    Args:
        course_id: Course ID to delete

    Returns:
        Deletion results
    """
    try:
        if not course_id or not course_id.strip():
            raise HTTPException(status_code=400, detail="Course ID cannot be empty")

        rag_service = await get_rag_service()
        deleted_count = await rag_service.delete_documents_by_course_id(course_id)

        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"No documents found for course_id: {course_id}")

        api_logger.info(f"Successfully deleted {deleted_count} documents for course_id: {course_id}")

        return {
            "message": f"Successfully deleted {deleted_count} documents for course_id: {course_id}",
            "course_id": course_id,
            "deletion_status": "completed"
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to delete course documents {course_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.delete("/material/{material_id}")
async def delete_material(material_id: str) -> Dict[str, Any]:
    """
    Delete all documents for a specific material_id

    Args:
        material_id: Material ID to delete

    Returns:
        Deletion results
    """
    try:
        if not material_id or not material_id.strip():
            raise HTTPException(status_code=400, detail="Material ID cannot be empty")

        rag_service = await get_rag_service()
        deleted_count = await rag_service.delete_documents_by_material_id(material_id)

        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"No documents found for material_id: {material_id}")

        api_logger.info(f"Successfully deleted {deleted_count} documents for material_id: {material_id}")

        return {
            "message": f"Successfully deleted {deleted_count} documents for material_id: {material_id}",
            "material_id": material_id,
            "deletion_status": "completed"
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to delete material {material_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post("/upload-templates")
async def upload_prompt_templates(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload CSV template containing prompt-response pairs for greetings, closings, and out-of-topics responses

    Expected CSV format:
    prompt,response,tag
    "Halo selamat pagi","Selamat pagi! Ada yang bisa saya bantu?",greeting
    "Terima kasih","Sama-sama! Senang bisa membantu Anda.",closing
    "Apakah cuaca?","Maaf, saya tidak memiliki informasi cuaca. Saya AI tutor untuk pembelajaran.",oot

    Args:
        file: CSV file with prompt-response pairs

    Returns:
        Processing results and statistics
    """
    try:
        # Validate file extension
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are allowed"
            )

        # Read and parse CSV with robust encoding handling
        content = await file.read()

        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        csv_content = None

        for encoding in encodings:
            try:
                csv_content = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if not csv_content:
            raise HTTPException(
                status_code=400,
                detail="Unable to decode CSV file. Please ensure it's saved as UTF-8"
            )

        # Remove BOM if present and clean up
        if csv_content.startswith('\ufeff'):
            csv_content = csv_content[1:]

        # Clean up the content and fix column names
        csv_content = csv_content.strip()

        csv_reader = csv.DictReader(io.StringIO(csv_content))

        # Validate required columns - be more flexible with column names
        available_columns = [col.strip().lower() for col in csv_reader.fieldnames] if csv_reader.fieldnames else []

        # Map various possible column names to standard names
        column_mapping = {}
        for col in available_columns:
            if 'prompt' in col.lower() or col.lower() == 'prompt':
                column_mapping['prompt'] = col
            elif 'response' in col.lower() or col.lower() == 'response':
                column_mapping['response'] = col
            elif 'tag' in col.lower():
                column_mapping['tag'] = col

        required_mappings = ['prompt', 'response', 'tag']
        missing_mappings = [mapping for mapping in required_mappings if mapping not in column_mapping]

        if missing_mappings:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must have columns containing: prompt, response, tag. Found columns: {available_columns}"
            )

        # Get cache service
        cache_service = await get_cache_service()

        processed_count = 0
        error_count = 0

        api_logger.info(f"Processing templates from {file.filename}")

        # Process each row
        for row in csv_reader:
            try:
                prompt = row[column_mapping['prompt']].strip()
                response = row[column_mapping['response']].strip()
                tag = row[column_mapping['tag']].strip().lower()

                # Skip empty rows
                if not prompt or not response:
                    continue

                # Validate tag
                valid_tags = {'greeting', 'closing', 'oot'}
                if tag not in valid_tags:
                    tag = 'oot'

                # Generate embedding and store
                embedding = await cache_service.generate_embedding(prompt)

                success = await cache_service.store_response(
                    prompt=prompt,
                    response=response,
                    embedding=embedding,
                    user_id="all",
                    model="gpt-4o-mini",
                    course_id="all",
                    sources=[{
                        "type": "template",
                        "tag": tag,
                        "source_file": file.filename
                    }]
                )

                if success:
                    processed_count += 1
                else:
                    error_count += 1

            except Exception:
                error_count += 1

        api_logger.info(f"Template upload completed: {processed_count} processed, {error_count} errors")

        return {
            "message": f"Successfully processed {processed_count} templates",
            "filename": file.filename,
            "processed_count": processed_count,
            "error_count": error_count,
            "total_rows": processed_count + error_count
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to upload templates: {e}")
        raise HTTPException(status_code=500, detail=f"Template upload failed: {str(e)}")
