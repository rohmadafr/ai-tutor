"""
Text Preprocessing Utilities for Document Processing
Simple and comprehensive text preprocessing for RAG knowledge base
"""
import re
import unicodedata
from typing import Optional
from app.core.logger import app_logger


class TextPreprocessor:
    """
    Simple text preprocessing utility that handles all common preprocessing tasks
    """

    def __init__(self):
        """Initialize text preprocessor with default settings"""
        # Compile regex patterns for performance
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+", flags=re.UNICODE
        )

        self.special_chars_pattern = re.compile(r"[^\w\s.,!?;:'\"()\-\n]")
        self.whitespace_pattern = re.compile(r"\s+")
        self.excessive_newlines_pattern = re.compile(r"\n{3,}")
        self.html_tag_pattern = re.compile(r"<[^>]+>")
        self.url_pattern = re.compile(r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?", flags=re.IGNORECASE)
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

        app_logger.info("TextPreprocessor initialized")

    def preprocess_text(self, text: str) -> str:
        """
        Apply comprehensive text preprocessing

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        try:
            # Unicode normalization
            text = unicodedata.normalize("NFKC", text)

            # Remove emojis
            text = self.emoji_pattern.sub(" ", text)

            # Remove HTML tags
            text = self.html_tag_pattern.sub(" ", text)

            # Remove URLs
            text = self.url_pattern.sub(" ", text)

            # Remove emails
            text = self.email_pattern.sub(" ", text)

            # Remove special characters (preserve basic punctuation)
            text = self.special_chars_pattern.sub(" ", text)

            # Handle excessive newlines (max 2 consecutive)
            text = self.excessive_newlines_pattern.sub("\n\n", text)

            # Normalize whitespace but preserve line breaks
            lines = text.split('\n')
            lines = [self.whitespace_pattern.sub(" ", line).strip() for line in lines]
            text = '\n'.join(lines)

            # Final cleanup
            text = text.strip()

            return text

        except Exception as e:
            app_logger.error(f"Error during text preprocessing: {e}")
            return text  # Return original text if preprocessing fails

    def preprocess_documents(self, documents: list) -> list:
        """
        Preprocess text for a list of document dictionaries

        Args:
            documents: List of document dicts with 'content' field

        Returns:
            List of preprocessed documents
        """
        processed_docs = []

        for i, doc in enumerate(documents):
            try:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                # Preprocess the content
                processed_content = self.preprocess_text(content)

                # Update metadata
                metadata.update({
                    "preprocessed": True,
                    "original_length": len(content),
                    "processed_length": len(processed_content)
                })

                processed_docs.append({
                    "content": processed_content,
                    "metadata": metadata
                })

                app_logger.debug(f"Preprocessed document {i+1}: {len(content)} -> {len(processed_content)} chars")

            except Exception as e:
                app_logger.error(f"Error preprocessing document {i+1}: {e}")
                # Keep original document if preprocessing fails
                processed_docs.append(doc)

        return processed_docs


# Global instance
text_preprocessor = TextPreprocessor()