import asyncio
from enum import Enum
from typing import List, Optional

from node import NodeWithScore

from rag.callbacks import CallbackManager
from rag.node.base_node import NodeWithScore

from .base import BaseRetriever
from .types import QueryBundle

class HYBRID_MODE(str, Enum):
    """Enum for different fusion modes."""

    OR = "or"  # get all 
    AND = "and"  # simple re-ordering of results based on original scores


class HybridSearchRetriever(BaseRetriever):
    """
    Custom retriever that uses both sparse and dense retrievers to retrieve nodes.
    """
    def __init__(
        self,
        sparse_retriever: BaseRetriever,  
        dense_retriever: BaseRetriever,  
        mode: HYBRID_MODE = HYBRID_MODE.AND,
        callback_manager: Optional[CallbackManager] = None,

    ) -> None:
        self._sparse_retriever = sparse_retriever
        self._dense_retriever = dense_retriever

        if mode not in ("and", "or"):
            raise ValueError("Invalid mode.")
        self._mode = mode

        super().__init__(
            callback_manager=callback_manager,
        )


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        
        vector_nodes = self._dense_retriever.retrieve(query_bundle)
        sparse_nodes = self._sparse_retriever.retrieve(query_bundle)

        vector_ids = {node.node.node_id for node in vector_nodes}
        sparse_ids = {node.node.node_id for node in sparse_nodes}

        combine_dict = {n.node.node_id: n for n in vector_nodes}
        combine_dict.update({n.node.node_id: n for n in sparse_nodes})

        if self._mode == HYBRID_MODE.AND:
            retrieve_ids = vector_ids.intersection(sparse_ids)
        else:
            retrieve_ids = vector_ids.union(sparse_ids)

        retrive_nodes = [combine_dict[rid] for rid in retrieve_ids]
        return retrive_nodes
    
    def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve(query_bundle)