"""Empty index.

An index that doesn't contain any documents. Can only be used for
pure LLM calls.

"""

from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

from rag.indices.data_struct import EmptyIndexStruct
from rag.indices.base_index import BaseIndex 

if TYPE_CHECKING:
    from rag.retrievers.base import BaseRetriever
    from rag.node.base_node import BaseNode
    from rag.core.service_context import ServiceContext
    from rag.storage.docstore.base import RefDocInfo


class EmptyIndex(BaseIndex):
    """Empty Index.

    An index that doesn't contain any documents. Used for
    pure LLM calls.
    NOTE: this exists because an empty index it allows certain properties,
    such as the ability to be composed with other indices + token
    counting + others.

    """

    def __init__(
        self,
        index_struct: Optional[EmptyIndexStruct] = None,
        service_context: Optional["ServiceContext"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            nodes=None,
            index_struct=index_struct or EmptyIndexStruct(),
            service_context=service_context,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> "BaseRetriever":
        # NOTE: lazy import
        from rag.retrievers.dense.emtpy_retriever import EmptyIndexRetriever

        return EmptyIndexRetriever(self)

    def _build_index_from_nodes(self, nodes: Sequence["BaseNode"]) -> EmptyIndexStruct:
        """Build the index from documents.

        Args:
            documents (List[BaseDocument]): A list of documents.

        Returns:
            IndexList: The created summary index.
        """
        del nodes  # Unused
        return EmptyIndexStruct()

    def _insert(self, nodes: Sequence["BaseNode"], **insert_kwargs: Any) -> None:
        """Insert a document."""
        del nodes  # Unused
        raise NotImplementedError("Cannot insert into an empty index.")

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError("Cannot delete from an empty index.")

    @property
    def ref_doc_info(self) -> Dict[str, "RefDocInfo"]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        raise NotImplementedError("ref_doc_info not supported for an empty index.")
