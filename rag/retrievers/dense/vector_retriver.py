"""Base vector store index query."""
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from rag.callbacks.callback_manager import CallbackManager
from rag.constants import DEFAULT_SIMILARITY_TOP_K
from rag.retrievers.base import BaseRetriever, QueryBundle
from rag.node.base_node import NodeWithScore, ObjectType
from rag.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from rag.indices.data_struct import IndexDict
from rag.indices.utils import log_vector_store_query_result

if TYPE_CHECKING:   
    from rag.indices.vector_store.base import VectorStoreIndex


class VectorIndexRetriever(BaseRetriever):
    """Vector index retriever.

    Args:
        index (VectorStoreIndex): vector store index.
        similarity_top_k (int): number of top k results to return.
        vector_store_query_mode (str): vector store query mode
            See reference for VectorStoreQueryMode for full list of supported modes.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        alpha (float): weight for sparse/dense retrieval, only used for
            hybrid query mode.
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        vector_store_kwargs (dict): Additional vector store specific kwargs to pass
            through to the vector store at query time.

    """

    def __init__(
        self,
        index: "VectorStoreIndex",
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        vector_store_query_mode: str = VectorStoreQueryMode.DEFAULT,
        filters: Optional[MetadataFilters] = None,
        alpha: Optional[float] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        sparse_top_k: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._vector_store = self._index.vector_store
        self._service_context = self._index.service_context
        self._docstore = self._index.docstore

        self._similarity_top_k = similarity_top_k
        self._vector_store_query_mode = vector_store_query_mode
        self._alpha = alpha
        self._node_ids = node_ids
        self._doc_ids = doc_ids
        self._filters = filters
        self._sparse_top_k = sparse_top_k
        self._kwargs: Dict[str, Any] = kwargs.get("vector_store_kwargs", {})
        super().__init__(callback_manager)

    @property
    def similarity_top_k(self) -> int:
        """Return similarity top k."""
        return self._similarity_top_k

    @similarity_top_k.setter
    def similarity_top_k(self, similarity_top_k: int) -> None:
        """Set similarity top k."""
        self._similarity_top_k = similarity_top_k

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                query_bundle.embedding = (
                    self._service_context.embed_model.get_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
        return self._get_nodes_with_embeddings(query_bundle)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self._vector_store.is_embedding_query:
            if query_bundle.embedding is None and len(query_bundle.embedding_strs) > 0:
                embed_model = self._service_context.embed_model
                query_bundle.embedding = (
                    await embed_model.aget_agg_embedding_from_queries(
                        query_bundle.embedding_strs
                    )
                )
        return await self._aget_nodes_with_embeddings(query_bundle)

    def _build_vector_store_query(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> VectorStoreQuery:
        return VectorStoreQuery(
            query_embedding=query_bundle_with_embeddings.embedding,
            similarity_top_k=self._similarity_top_k,
            node_ids=self._node_ids,
            doc_ids=self._doc_ids,
            query_str=query_bundle_with_embeddings.query_str,
            mode=self._vector_store_query_mode,
            alpha=self._alpha,
            filters=self._filters,
            sparse_top_k=self._sparse_top_k,
        )

    def _build_node_list_from_query_result(
        self, query_result: VectorStoreQueryResult
    ) -> List[NodeWithScore]:
        if query_result.nodes is None:
            # NOTE: vector store does not keep text and returns node indices.
            # Need to recover all nodes from docstore
            if query_result.ids is None:
                raise ValueError(
                    "Vector store query result should return at "
                    "least one of nodes or ids."
                )
            assert isinstance(self._index.index_struct, IndexDict)
            node_ids = [
                self._index.index_struct.nodes_dict[idx] for idx in query_result.ids
            ]
            nodes = self._docstore.get_nodes(node_ids)
            query_result.nodes = nodes
        else:
            # NOTE: vector store keeps text, returns nodes.
            # Only need to recover image or index nodes from docstore
            for i in range(len(query_result.nodes)):
                source_node = query_result.nodes[i].source_node
                if (not self._vector_store.stores_text) or (
                    source_node is not None and source_node.node_type != ObjectType.TEXT
                ):
                    node_id = query_result.nodes[i].node_id
                    if self._docstore.document_exists(node_id):
                        query_result.nodes[
                            i
                        ] = self._docstore.get_node(  # type: ignore[index]
                            node_id
                        )

        log_vector_store_query_result(query_result)

        node_with_scores: List[NodeWithScore] = []
        for ind, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[ind]
            node_with_scores.append(NodeWithScore(node=node, score=score))

        return node_with_scores

    def _get_nodes_with_embeddings(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(query_bundle_with_embeddings)
        query_result = self._vector_store.query(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result)

    async def _aget_nodes_with_embeddings(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(query_bundle_with_embeddings)
        query_result = await self._vector_store.aquery(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result)