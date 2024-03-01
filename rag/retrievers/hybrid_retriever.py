import asyncio
from enum import Enum
from typing import Dict, List, Optional, Tuple, cast

from rag.rag_utils.async_utils import run_async_tasks
from rag.callbacks import CallbackManager
from rag.constants import DEFAULT_SIMILARITY_TOP_K
from rag.llm.resolve_llm import resolve_llm
from rag.prompt.prompt_template import PromptTemplate
from rag.prompt.mixin import PromptDictType
from rag.retrievers.base_retriver import BaseRetriever
from rag.retrievers.types import QueryBundle
from rag.node.base_node import IndexNode, NodeWithScore
from rag.llm.base import LLMType

QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)


class HYBRID_MODE(str, Enum):
    """Enum for different fusion modes."""

    OR = "or"  # apply reciprocal rank fusion
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
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,

    ) -> None:
        self._similarity_top_k = similarity_top_k
        self._sparse_retriever = sparse_retriever
        self._dense_retriever = dense_retriever
        self._similarity_top_k = similarity_top_k

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
        return retrive_nodes[: self._similarity_top_k]