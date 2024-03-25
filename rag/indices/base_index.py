"""Base index classes."""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from rag.callbacks import CallbackManager
from rag.indices.utils import run_transformations
from rag.node.base_node import BaseNode, Document
from rag.core.storage_context import StorageContext
from rag.core.service_context import ServiceContext

from .data_struct import IndexStruct

if TYPE_CHECKING:
    from retrievers.base import BaseRetriever
    from rag.storage.docstore.base import BaseDocumentStore, RefDocInfo



logger = logging.getLogger(__name__)


class BaseIndex(ABC):
    """Base Index.

    Args:
        nodes (List[Node]): List of nodes to index
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        storage_context (StorageContext): Storage context container (contains
            storage to persist like docstore, index_store, etc.).

    """

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[IndexStruct] = None,
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None and nodes is None:
            raise ValueError("`nodes` or `index_struct` must be provided.")
        # This is to explicitly make sure that the old UX is not used
        if nodes is not None and len(nodes) >= 1 and not isinstance(nodes[0], BaseNode):
            if isinstance(nodes[0], Document):
                raise ValueError(
                    "The constructor now takes in a list of Node objects. "
                    "Since you are passing in a list of Document objects, "
                    "please use `from_documents` instead."
                )
            else:
                raise ValueError("nodes must be a list of Node objects.")

        self._service_context = service_context or ServiceContext()
        self._storage_context = storage_context or StorageContext.from_defaults()
        self._docstore = self._storage_context.docstore
        self._callback_manager = self._service_context.callback_manager
        self._show_progress = show_progress

        with self._callback_manager.as_trace("index_construction"):
            if index_struct is None:
                assert nodes is not None
                index_struct = self.build_index_from_nodes(nodes)
                
            self._index_struct = index_struct

        self._storage_context.index_store.add_index_struct(self._index_struct)
            

    @property
    def index_struct(self) -> IndexStruct:
        """Get the index struct."""
        return self._index_struct

    @property
    def index_id(self) -> str:
        """Get the index struct."""
        return self.index_struct.index_id

    @property
    def docstore(self) -> "BaseDocumentStore":
        """Get the docstore corresponding to the index."""
        return self._docstore

    @property
    def service_context(self) -> "ServiceContext":
        return self._service_context

    @property
    def storage_context(self) -> "StorageContext":
        return self._storage_context
    
    
    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        callback_manager: Optional[CallbackManager] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> "BaseIndex":
        """Create index from documents.

        Args:
            documents (Optional[Sequence[BaseDocument]]): List of documents to
                build the index from.

        """
        storage_context = storage_context or StorageContext.from_defaults()
        service_context = service_context or ServiceContext()
        docstore = storage_context.docstore
        callback_manager = service_context.callback_manager
        transformations = service_context.transformations 

        with callback_manager.as_trace("index_construction"):
            for doc in documents:
                docstore.set_document_hash(doc.get_doc_id(), doc.hash)

            nodes = run_transformations(
                documents,  # type: ignore
                transformations,
                show_progress=show_progress,
                **kwargs,
            )

            return cls(
                nodes=nodes,
                storage_context=storage_context,
                callback_manager=callback_manager,
                show_progress=show_progress,
                **kwargs,
            )


    @abstractmethod
    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexStruct:
        """Build the index from nodes."""


    def build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexStruct:
        """Build the index from nodes."""
        self.docstore.add_documents(nodes, allow_update=True)
        return self._build_index_from_nodes(nodes)


    @abstractmethod
    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""


    def insert_nodes(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert nodes."""
        with self._callback_manager.as_trace("insert_nodes"):
            self.docstore.add_documents(nodes, allow_update=True)
            self._insert(nodes, **insert_kwargs)
            self.storage_context.index_store.add_index_struct(self.index_struct)


    def insert_document(self, document: Document, **insert_kwargs: Any) -> None:
        """Insert a document."""
        with self._callback_manager.as_trace("insert"):
            nodes = run_transformations(
                [document],
                self._service_context.transformations,
                show_progress=self._show_progress,
            )

            self.insert_nodes(nodes, **insert_kwargs)
            self.docstore.set_document_hash(document.get_doc_id(), document.hash)


    @abstractmethod
    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""


    def delete_nodes(
        self,
        node_ids: List[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        """Delete a list of nodes from the index.

        Args:
            doc_ids (List[str]): A list of doc_ids from the nodes to delete

        """
        for node_id in node_ids:
            self._delete_node(node_id, **delete_kwargs)
            if delete_from_docstore:
                self.docstore.delete_document(node_id, raise_error=False)

        self.storage_context.index_store.add_index_struct(self.index_struct)


    def delete_ref_doc(
        self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any
    ) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        ref_doc_info = self.docstore.get_ref_doc_info(ref_doc_id)
        if ref_doc_info is None:
            logger.warning(f"ref_doc_id {ref_doc_id} not found, nothing deleted.")
            return

        self.delete_nodes(
            ref_doc_info.node_ids,
            delete_from_docstore=False,
            **delete_kwargs,
        )

        if delete_from_docstore:
            self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)


    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        """Update a document and it's corresponding nodes.

        This is equivalent to deleting the document and then inserting it again.

        Args:
            document (Union[BaseDocument, BaseIndex]): document to update
            insert_kwargs (Dict): kwargs to pass to insert
            delete_kwargs (Dict): kwargs to pass to delete

        """
        with self._callback_manager.as_trace("update"):
            self.delete_ref_doc(
                document.get_doc_id(),
                delete_from_docstore=True,
                **update_kwargs.pop("delete_kwargs", {}),
            )
            self.insert_document(document, **update_kwargs.pop("insert_kwargs", {}))


    def refresh_ref_docs(
        self, documents: Sequence[Document], **update_kwargs: Any
    ) -> List[bool]:
        """Refresh an index with documents that have changed.

        This allows users to save LLM and Embedding model calls, while only
        updating documents that have any changes in text or metadata. It
        will also insert any documents that previously were not stored.
        """
        with self._callback_manager.as_trace("refresh"):
            refreshed_documents = [False] * len(documents)
            for i, document in enumerate(documents):
                existing_doc_hash = self.docstore.get_document_hash(
                    document.get_doc_id()
                )
                if existing_doc_hash is None:
                    self.insert_document(document, **update_kwargs.pop("insert_kwargs", {}))
                    refreshed_documents[i] = True
                elif existing_doc_hash != document.hash:
                    self.update_ref_doc(
                        document, **update_kwargs.pop("update_kwargs", {})
                    )
                    refreshed_documents[i] = True

            return refreshed_documents

    @property
    @abstractmethod
    def ref_doc_info(self) -> Dict[str, "RefDocInfo"]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        ...

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> "BaseRetriever":
        ...
