"""Default query for EmptyIndex."""
from typing import Any, List, Optional, TYPE_CHECKING

from rag.entity.callbacks.callback_manager import CallbackManager
from rag.entity.retriever.base_retriver import BaseRetriever, QueryBundle
from rag.constants.default_prompt import DEFAULT_SIMPLE_INPUT_PROMPT
from .base import EmptyIndex

if TYPE_CHECKING:
    from rag.entity.prompt import BasePromptTemplate
    from rag.entity.node import NodeWithScore


class EmptyIndexRetriever(BaseRetriever):
    """EmptyIndex query.

    Passes the raw LLM call to the underlying LLM model.

    Args:
        input_prompt (Optional[BasePromptTemplate]): A Simple Input Prompt
            (see :ref:`Prompt-Templates`).

    """

    def __init__(
        self,
        index: EmptyIndex,
        input_prompt: Optional["BasePromptTemplate"] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._input_prompt = input_prompt or DEFAULT_SIMPLE_INPUT_PROMPT
        super().__init__(callback_manager)

    def _retrieve(self, query_bundle: "QueryBundle") -> List["NodeWithScore"]:
        """Retrieve relevant nodes."""
        del query_bundle  # Unused
        return []