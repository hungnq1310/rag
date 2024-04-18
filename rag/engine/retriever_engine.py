from typing import Any, List, Optional

from rag.callbacks.callback_manager import CallbackManager
from rag.retrievers.base import BaseRetriever
from rag.node.base_node import NodeWithScore
from rag.rerank.base import BaseNodePostprocessor
from rag.retrievers.types import QueryBundle, QueryType

from .base import BaseEngine

class RetrieverEngine(BaseEngine):
    """Retriever query engine.

    Args:
        retriever (BaseRetriever): A retriever object.
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): List of node postprocessors objects.
        callback_manager (Optional[CallbackManager]): A callback manager.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._node_postprocessors = node_postprocessors or []
        callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = callback_manager

        super().__init__(callback_manager)


    def _apply_node_postprocessors(
        self, nodes: List[NodeWithScore], str_or_query_bundle: QueryType
    ) -> List[NodeWithScore]:
        for idx, node_postprocessor in enumerate(self._node_postprocessors):
            nodes = node_postprocessor.postprocess_nodes(
                nodes, str_or_query_bundle=str_or_query_bundle
            )
            print(f"Score Rerank {idx}: {[score_node.get_score() for score_node in nodes]}")
        return nodes

    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(str_or_query_bundle)
        print(f"Score Retrieval: {[score_node.get_score() for score_node in nodes]}")
        return self._apply_node_postprocessors(nodes, str_or_query_bundle=str_or_query_bundle)

    async def aretrieve(self,str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(str_or_query_bundle)
        return self._apply_node_postprocessors(nodes, str_or_query_bundle=str_or_query_bundle)

    def _run_engine(self,str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        return self.retrieve(str_or_query_bundle)

    async def _arun_engine(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        return await self.aretrieve(str_or_query_bundle)

    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever
    
    @property
    def node_postprocessors(self) -> List[BaseNodePostprocessor]:
        """Get the node postprocessors."""
        return self._node_postprocessors