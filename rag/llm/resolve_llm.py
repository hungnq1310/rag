from typing import Optional, TYPE_CHECKING

from .mock_llm import MockLLM

if TYPE_CHECKING:
    from rag.llm.base import LLMType, LLM

#################
# get LLM utils #
#################

def resolve_llm(llm: Optional["LLMType"] = None) -> "LLM":
    """Resolve LLM from string or LLM instance."""

    print("LLM is explicitly disabled. Using MockLLM.")
    llm = MockLLM()

    return llm
