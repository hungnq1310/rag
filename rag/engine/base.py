"""Base query engine."""

import logging
from abc import abstractmethod, ABC
from typing import Any, Optional

from rag.callbacks.callback_manager import CallbackManager
from rag.retrievers.types import QueryBundle, QueryType

logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """Base query engine."""

    def __init__(self, callback_manager: Optional[CallbackManager]) -> None:
        self.callback_manager = callback_manager or CallbackManager([])

    def __call__(self, str_or_query_bundle: QueryType) -> Any:
        with self.callback_manager.as_trace("run engine"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return self._run_engine(str_or_query_bundle)

    async def __acall__(self, str_or_query_bundle: QueryType) -> Any:
        with self.callback_manager.as_trace("run async engine"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            return await self._arun_engine(str_or_query_bundle)

    @abstractmethod
    def _run_engine(self, str_or_query_bundle: QueryType) -> Any:
        pass

    @abstractmethod
    async def _arun_engine(self, str_or_query_bundle: QueryType) -> Any:
        pass
