"""Empty index.

An index that doesn't contain any documents. Can only be used for
pure LLM calls.

"""

from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

from ..data_struct import EmptyIndexStruct
from ..base_index import BaseIndex 

if TYPE_CHECKING:
    from rag.entity.retriever import BaseRetriever
    from rag.entity.base_query_engine import BaseQueryEngine
    from rag.entity.node import BaseNode
    from rag.entity.service_context import ServiceContext
    from rag.entity.storage.docstore import RefDocInfo


class EmptyIndex(BaseIndex[EmptyIndexStruct]):
    """Empty Index.

    An index that doesn't contain any documents. Used for
    pure LLM calls.
    NOTE: this exists because an empty index it allows certain properties,
    such as the ability to be composed with other indices + token
    counting + others.

    """

    index_struct_cls = EmptyIndexStruct

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
        from rag.entity.indices.empty import EmptyIndexRetriever

        return EmptyIndexRetriever(self)

    def as_query_engine(self, **kwargs: Any) -> "BaseQueryEngine":
        if "response_mode" not in kwargs:
            kwargs["response_mode"] = "generation"
        else:
            if kwargs["response_mode"] != "generation":
                raise ValueError("EmptyIndex only supports response_mode=generation.")

        return super().as_query_engine(**kwargs)

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
