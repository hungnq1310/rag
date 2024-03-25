"""Base retriever."""
from abc import abstractmethod, ABC
from typing import List, Optional

from rag.callbacks import CallbackManager, CBEventType, EventPayload
from rag.node import NodeWithScore
from .types import *


class BaseRetriever(ABC):
    """Base retriever."""

    def __init__(self, callback_manager: Optional[CallbackManager] = None) -> None:
        self.callback_manager = callback_manager or CallbackManager()

    def _check_callback_manager(self) -> None:
        """Check callback manager."""
        if not hasattr(self, "callback_manager"):
            self.callback_manager = CallbackManager()


    def retrieve(self, str_or_query_bundle: Union[str, QueryBundle]) -> List["NodeWithScore"]:
        """Retrieve nodes given query.

        Args:
            str_or_query_bundle (QueryType): Either a query string or
                a QueryBundle object.

        """
        self._check_callback_manager()

        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self._retrieve(query_bundle)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        return nodes

    async def aretrieve(self, str_or_query_bundle: Union[str, QueryBundle]) -> List["NodeWithScore"]:
        self._check_callback_manager()

        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(query_bundle)
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        return nodes

    @abstractmethod
    def _retrieve(self, query_bundle: QueryBundle) -> List["NodeWithScore"]:
        """Retrieve nodes given query.

        Implemented by the user.

        """

    # TODO: make this abstract
    # @abstractmethod
    async def _aretrieve(self, query_bundle: QueryBundle) -> List["NodeWithScore"]:
        """Asynchronously retrieve nodes given query.

        Implemented by the user.

        """
        return self._retrieve(query_bundle)