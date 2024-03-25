from typing import Any, List, Optional

from rag.callbacks.callback_manager import CallbackManager
from retrievers.base import BaseRetriever, QueryBundle
from rag.node.base_node import NodeWithScore
from rag.rerank.base import BaseNodePostprocessor

from .base import BaseEngine

class RetrieverEngine(BaseEngine):
    """Retriever query engine.

    Args:
        retriever (BaseRetriever): A retriever object.
        response_synthesizer (Optional[BaseSynthesizer]): A BaseSynthesizer
            object.
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
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = await self._retriever.aretrieve(query_bundle)
        return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    def _run_engine(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self.retrieve(query_bundle)

    async def _arun_engine(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return await self.aretrieve(query_bundle)

    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever