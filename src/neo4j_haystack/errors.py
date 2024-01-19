from typing import Optional

from haystack.document_stores.errors import DocumentStoreError


class Neo4jDocumentStoreError(DocumentStoreError):
    """Error for issues that occur in a Neo4j Document Store"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message)


class Neo4jClientError(DocumentStoreError):
    """Error for issues that occur in a Neo4j client"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message)


class Neo4jFilterParserError(Exception):
    """Error is raised when metadata filters are failing to parse"""

    pass
