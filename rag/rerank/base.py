from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING


from rag.bridge.pydantic import Field
from rag.callbacks import CallbackManager
from rag.schema.component import BaseComponent
from rag.retrievers.types import QueryBundle
from rag.node.base_node import NodeWithScore

if TYPE_CHECKING:
    from rag.prompt.mixin import PromptDictType, PromptMixinType


class BaseNodePostprocessor(BaseComponent, ABC):
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    class Config:
        arbitrary_types_allowed = True

    def _get_prompts(self) -> "PromptDictType":
        """Get prompts."""
        # set by default since most postprocessors don't require prompts
        return {}

    def _update_prompts(self, prompts: "PromptDictType") -> None:
        """Update prompts."""

    def _get_prompt_modules(self) -> "PromptMixinType":
        """Get prompt modules."""
        return {}

    # implement class_name so users don't have to worry about it when extending
    @classmethod
    def class_name(cls) -> str:
        return "BaseNodePostprocessor"

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        if query_str is not None and query_bundle is not None:
            raise ValueError("Cannot specify both query_str and query_bundle")
        elif query_str is not None:
            query_bundle = QueryBundle(query_str)
        else:
            pass
        return self._postprocess_nodes(nodes, query_bundle)

    @abstractmethod
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""