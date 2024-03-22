"""Retriever tool."""

from typing import TYPE_CHECKING, Any, Optional

from rag.node.base_node import MetadataMode

from .base import AsyncBaseTool
from .types import ToolMetadata, ToolOutput

if TYPE_CHECKING:
    from rag.retrievers.base_retriver import BaseRetriever

DEFAULT_NAME = "retriever_tool"
DEFAULT_DESCRIPTION = """Useful for running a natural language query
against a knowledge base and retrieving a set of relevant documents.
"""


class RetrieverTool(AsyncBaseTool):
    """Retriever tool.

    A tool making use of a retriever.

    Args:
        retriever (BaseRetriever): A retriever.
        metadata (ToolMetadata): The associated metadata of the query engine.
    """

    def __init__(
        self,
        retriever: "BaseRetriever",
        metadata: ToolMetadata,
    ) -> None:
        self._retriever = retriever
        self._metadata = metadata

    @classmethod
    def from_defaults(
        cls,
        retriever: "BaseRetriever",
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "RetrieverTool":
        name = name or DEFAULT_NAME
        description = description or DEFAULT_DESCRIPTION

        metadata = ToolMetadata(name=name, description=description)
        return cls(retriever=retriever, metadata=metadata)

    @property
    def retriever(self) -> "BaseRetriever":
        return self._retriever

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        query_str = ""
        if args is not None:
            query_str += ", ".join([str(arg) for arg in args]) + "\n"
        if kwargs is not None:
            query_str += (
                ", ".join([f"{k!s} is {v!s}" for k, v in kwargs.items()]) + "\n"
            )
        if query_str == "":
            raise ValueError("Cannot call query engine without inputs")

        docs = self._retriever.retrieve(query_str)
        content = ""
        for doc in docs:
            node_copy = doc.node.copy()
            node_copy.text_template = "{metadata_str}\n{content}"
            node_copy.metadata_template = "{key} = {value}"
            content += node_copy.get_content(MetadataMode.LLM) + "\n\n"
        return ToolOutput(
            content=content,
            tool_name=self.metadata.name,
            raw_input={"input": input},
            raw_output=docs,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        query_str = ""
        if args is not None:
            query_str += ", ".join([str(arg) for arg in args]) + "\n"
        if kwargs is not None:
            query_str += (
                ", ".join([f"{k!s} is {v!s}" for k, v in kwargs.items()]) + "\n"
            )
        if query_str == "":
            raise ValueError("Cannot call query engine without inputs")
        docs = await self._retriever.aretrieve(query_str)
        content = ""
        for doc in docs:
            node_copy = doc.node.copy()
            node_copy.text_template = "{metadata_str}\n{content}"
            node_copy.metadata_template = "{key} = {value}"
            content += node_copy.get_content(MetadataMode.LLM) + "\n\n"
        return ToolOutput(
            content=content,
            tool_name=self.metadata.name,
            raw_input={"input": input},
            raw_output=docs,
        )