"""Vector store index types."""
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)
import fsspec

from rag.entity.node.base_node import BaseNode
from .types import *

@runtime_checkable
class VectorStore(Protocol):
    """Abstract vector store protocol."""

    stores_text: bool
    is_embedding_query: bool = True

    @property
    def client(self) -> Any:
        """Get client."""
        ...

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes with embedding to vector store."""
        ...

    async def async_add(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Asynchronously add nodes with embedding to vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call add synchronously.
        """
        return self.add(nodes)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id."""
        ...

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call delete synchronously.
        """
        self.delete(ref_doc_id, **delete_kwargs)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        ...

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Asynchronously query vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call query synchronously.
        """
        return self.query(query, **kwargs)

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        return None