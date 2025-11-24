# File: app/utils/material_ider.py
"""
Utility untuk menghasilkan hash file dan tanda tangan konten.
(Dipindah dari utils.py)
"""
import hashlib
import time
from pathlib import Path

# Impor dari struktur baru
from app.core.logger import app_logger
from app.core.exceptions import FileProcessingError

class FileHasher:
    """Utility for generating file hashes and content signatures."""

    @staticmethod
    def get_material_id(file_path: Path, algorithm: str = "md5") -> str:
        """
        Generate hash of file contents.
        """
        try:
            hash_func = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            app_logger.error(f"Error hashing file {file_path}: {e}")
            raise FileProcessingError(f"Could not hash file: {e}")

    @staticmethod
    def get_content_signature(file_path: Path) -> str:
        """
        Generate content signature based on file size, mtime, and partial hash.
        """
        try:
            stat = file_path.stat()
            size = stat.st_size
            mtime = stat.st_mtime

            if size < 1024 * 1024:
                return FileHasher.get_material_id(file_path, "md5")

            hash_func = hashlib.md5()
            with open(file_path, 'rb') as f:
                first_chunk = f.read(4096)
                hash_func.update(first_chunk)
                hash_func.update(f"{size}_{mtime}".encode())
                if size > 8192:
                    f.seek(-4096, 2)
                    last_chunk = f.read(4096)
                    hash_func.update(last_chunk)
            return f"sig_{hash_func.hexdigest()[:16]}"
        except Exception as e:
            app_logger.error(f"Error generating content signature for {file_path}: {e}")
            return f"fallback_{hash(f'{file_path}_{time.time()}')}"

    