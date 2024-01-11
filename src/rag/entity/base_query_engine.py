"""Base query engine."""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from rag.entity.callbacks import CallbackManager
from rag.entity.synthesizer import RESPONSE_TYPE
from rag.components.prompt import PromptDictType, PromptMixin
from rag.entity.node import NodeWithScore
from rag.entity.retriever import QueryBundle, QueryType

logger = logging.getLogger(__name__)


class BaseQueryEngine(PromptMixin):
    """Base query engine."""

    def __init__(self, callback_manager: Optional[CallbackManager]) -> None:
        self.callback_manager = callback_manager or CallbackManager([])

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return self._query(str_or_query_bundle)

    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return await self._aquery(str_or_query_bundle)

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError(
            "This query engine does not support retrieve, use query directly"
        )

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        raise NotImplementedError(
            "This query engine does not support synthesize, use query directly"
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        raise NotImplementedError(
            "This query engine does not support asynthesize, use aquery directly"
        )

    @abstractmethod
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass

    @abstractmethod
    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        pass
