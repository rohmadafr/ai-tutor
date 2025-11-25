"""
Document Management API for RAG Knowledge Base
Handles document upload, ingestion, search, and deletion operations
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, Optional
import tempfile
from pathlib import Path

from ..services.unified_rag_service import UnifiedRAGService
from ..utils.pdf_extractor import PDFExtractor
from ..utils.file_hasher import FileHasher
from ..utils.text_preprocessing import text_preprocessor
from ..core.logger import api_logger

router = APIRouter(prefix="/documents", tags=["documents"])

# Global service instances
_rag_service: Optional[UnifiedRAGService] = None


async def get_rag_service() -> UnifiedRAGService:
    """Get RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = UnifiedRAGService()
    return _rag_service


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

            api_logger.info(f"Successfully processed {file.filename}: {len(documents)} chunks, material_id: {material_id}")

            return {
                "message": f"Successfully uploaded and processed {file.filename}",
                "filename": file.filename,
                "material_id": material_id,
                "course_id": course_id,
                "total_chunks": len(documents),
                "file_type": file_extension,
                "preprocessed": True,
                "total_chars": sum(len(doc["content"]) for doc in documents)
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
