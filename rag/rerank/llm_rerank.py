"""LLM reranker."""
from typing import Callable, List, Optional, TYPE_CHECKING

from rag.bridge.pydantic import Field, PrivateAttr
from rag.indices.utils import (
    default_format_node_batch_fn,
    default_parse_choice_select_answer_fn,
)
from rag.constants.default_prompt import DEFAULT_CHOICE_SELECT_PROMPT
from rag.node.base_node import NodeWithScore
from rag.retrievers.types import QueryBundle

from .base import BaseNodePostprocessor

if TYPE_CHECKING:
    from rag.llm.base import LLM
    from rag.core.service_context import ServiceContext
    from rag.prompt.base_prompt import BasePromptTemplate
    from rag.prompt.mixin import PromptDictType

class LLMRerank(BaseNodePostprocessor):
    """LLM-based reranker."""

    top_n: int = Field(description="Top N nodes to return.")
    choice_select_prompt: "BasePromptTemplate" = Field(
        description="Choice select prompt."
    )
    choice_batch_size: int = Field(description="Batch size for choice select.")
    llm: "LLM" = Field(description="The LLM to rerank with.")

    _format_node_batch_fn: Callable = PrivateAttr()
    _parse_choice_select_answer_fn: Callable = PrivateAttr()

    def __init__(
        self,
        llm: Optional["LLM"] = None,
        choice_select_prompt: Optional["BasePromptTemplate"] = None,
        choice_batch_size: int = 10,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        service_context: Optional["ServiceContext"] = None,
        top_n: int = 10,
    ) -> None:
        choice_select_prompt = choice_select_prompt or DEFAULT_CHOICE_SELECT_PROMPT

        llm = llm

        self._format_node_batch_fn = (
            format_node_batch_fn or default_format_node_batch_fn
        )
        self._parse_choice_select_answer_fn = (
            parse_choice_select_answer_fn or default_parse_choice_select_answer_fn
        )

        super().__init__(
            llm=llm,
            choice_select_prompt=choice_select_prompt,
            choice_batch_size=choice_batch_size,
            service_context=service_context,
            top_n=top_n,
        )

    def _get_prompts(self) -> "PromptDictType":
        """Get prompts."""
        return {"choice_select_prompt": self.choice_select_prompt}

    def _update_prompts(self, prompts: "PromptDictType") -> None:
        """Update prompts."""
        if "choice_select_prompt" in prompts:
            self.choice_select_prompt = prompts["choice_select_prompt"]

    @classmethod
    def class_name(cls) -> str:
        return "LLMRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        initial_results: List[NodeWithScore] = []
        for idx in range(0, len(nodes), self.choice_batch_size):
            nodes_batch = [
                node.node for node in nodes[idx : idx + self.choice_batch_size]
            ]

            query_str = query_bundle.query_str
            fmt_batch_str = self._format_node_batch_fn(nodes_batch)
            # call each batch independently
            raw_response = self.llm.predict(
                self.choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str,
            )

            raw_choices, relevances = self._parse_choice_select_answer_fn(
                raw_response, len(nodes_batch)
            )
            choice_idxs = [int(choice) - 1 for choice in raw_choices]
            choice_nodes = [nodes_batch[idx] for idx in choice_idxs]
            relevances = relevances or [1.0 for _ in choice_nodes]
            initial_results.extend(
                [
                    NodeWithScore(node=node, score=relevance)
                    for node, relevance in zip(choice_nodes, relevances)
                ]
            )

        return sorted(initial_results, key=lambda x: x.score or 0.0, reverse=True)[
            : self.top_n
        ]