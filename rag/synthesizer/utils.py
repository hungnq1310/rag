from typing import Optional, TYPE_CHECKING

from rag.bridge.pydantic import BaseModel
# TODO: explore selectors
from rag.prompt.selector_template import (
    DEFAULT_TEXT_QA_PROMPT_SEL,
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)
from rag.constants.default_prompt import DEFAULT_SIMPLE_INPUT_PROMPT
from rag.core.service_context import ServiceContext
from .mode import ResponseMode
from .generation import Generation
from .no_text import NoText
from .summary import SimpleSummarize
from .tree_summary import TreeSummarize

if TYPE_CHECKING:
    from rag.callbacks.callback_manager import CallbackManager
    from rag.prompt import BasePromptTemplate
    from rag.synthesizer.base_synthesizer import BaseSynthesizer


def get_response_synthesizer(
    service_context: Optional[ServiceContext] = None,
    text_qa_template: Optional["BasePromptTemplate"] = None,
    summary_template: Optional["BasePromptTemplate"] = None,
    simple_template: Optional["BasePromptTemplate"] = None,
    response_mode: ResponseMode = ResponseMode.SIMPLE_SUMMARIZE,
    callback_manager: Optional["CallbackManager"] = None,
    use_async: bool = False,
    streaming: bool = False,
    output_cls: Optional[BaseModel] = None,
    verbose: bool = False,
) -> "BaseSynthesizer":
    """Get a response synthesizer."""
    text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
    simple_template = simple_template or DEFAULT_SIMPLE_INPUT_PROMPT
    summary_template = summary_template or DEFAULT_TREE_SUMMARIZE_PROMPT_SEL

    service_context = service_context or ServiceContext.from_defaults(
        callback_manager=callback_manager
    )
    if response_mode == ResponseMode.TREE_SUMMARIZE:
        return TreeSummarize(
            service_context=service_context,
            summary_template=summary_template,
            output_cls=output_cls,
            streaming=streaming,
            use_async=use_async,
            verbose=verbose,
        )
    elif response_mode == ResponseMode.SIMPLE_SUMMARIZE:
        return SimpleSummarize(
            service_context=service_context,
            text_qa_template=text_qa_template,
            streaming=streaming,
        )
    elif response_mode == ResponseMode.GENERATION:
        return Generation(
            service_context=service_context,
            simple_template=simple_template,
            streaming=streaming,
        )
    elif response_mode == ResponseMode.NO_TEXT:
        return NoText(
            service_context=service_context,
            streaming=streaming,
        )
    else:
        raise ValueError(f"Unknown mode: {response_mode}")