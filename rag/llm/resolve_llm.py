from typing import Optional, TYPE_CHECKING

from .llama_cpp import LlamaCPP
from .mock_llm import MockLLM
from .utils import messages_to_prompt, completion_to_prompt

if TYPE_CHECKING:
    from rag.llm.base import LLMType, LLM

#################
# get LLM utils #
#################

def resolve_llm(llm: Optional["LLMType"] = None) -> "LLM":
    """Resolve LLM from string or LLM instance."""

    if isinstance(llm, str) or llm == "default":
        splits = llm.split(":", 1)
        is_local = splits[0]
        model_path = splits[1] if len(splits) > 1 else None
        if is_local != "local":
            raise ValueError(
                "llm must start with str 'local' or of type LLM or BaseLanguageModel"
            )
        llm = LlamaCPP(
            model_path=model_path,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            model_kwargs={"n_gpu_layers": 1},
        )

    elif llm is None:
        print("LLM is explicitly disabled. Using MockLLM.")
        llm = MockLLM()

    return llm
