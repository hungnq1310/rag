"""Prompts."""
from typing import Any, Callable, Tuple, List, Optional, TYPE_CHECKING
import logging

from .base_prompt import BasePromptTemplate 
from .utils import is_chat_model

if TYPE_CHECKING:
    from rag.llm.base import LLM, ChatMessage

logger = logging.getLogger(__name__)

class SelectorPromptTemplate(BasePromptTemplate):
    default_template: BasePromptTemplate
    conditionals: Optional[
        List[Tuple[Callable[["LLM"], bool], BasePromptTemplate]]
    ] = None

    def __init__(
        self,
        default_template: BasePromptTemplate,
        conditionals: Optional[
            List[Tuple[Callable[["LLM"], bool], BasePromptTemplate]]
        ] = None,
    ):
        metadata = default_template.metadata
        kwargs = default_template.kwargs
        template_vars = default_template.template_vars
        output_parser = default_template.output_parser
        super().__init__(
            default_template=default_template,
            conditionals=conditionals,
            metadata=metadata,
            kwargs=kwargs,
            template_vars=template_vars,
            output_parser=output_parser,
        )

    def select(self, llm: Optional["LLM"] = None) -> BasePromptTemplate:
        # ensure output parser is up to date
        self.default_template.output_parser = self.output_parser

        if llm is None:
            return self.default_template

        if self.conditionals is not None:
            for condition, prompt in self.conditionals:
                if condition(llm):
                    # ensure output parser is up to date
                    prompt.output_parser = self.output_parser
                    return prompt

        return self.default_template

    def partial_format(self, **kwargs: Any) -> "SelectorPromptTemplate":
        default_template = self.default_template.partial_format(**kwargs)
        if self.conditionals is None:
            conditionals = None
        else:
            conditionals = [
                (condition, prompt.partial_format(**kwargs))
                for condition, prompt in self.conditionals
            ]
        return SelectorPromptTemplate(
            default_template=default_template, conditionals=conditionals
        )

    def format(self, llm: Optional["LLM"] = None, **kwargs: Any) -> str:
        """Format the prompt into a string."""
        prompt = self.select(llm=llm)
        return prompt.format(**kwargs)

    def format_messages(
        self, llm: Optional["LLM"] = None, **kwargs: Any
    ) -> List["ChatMessage"]:
        """Format the prompt into a list of chat messages."""
        prompt = self.select(llm=llm)
        return prompt.format_messages(**kwargs)

    def get_template(self, llm: Optional["LLM"] = None) -> str:
        prompt = self.select(llm=llm)
        return prompt.get_template(llm=llm)
    

"""Default prompt selectors."""
from .chat_template import (
    CHAT_TEXT_QA_PROMPT,
    CHAT_TREE_SUMMARIZE_PROMPT,
)
from rag.constants.default_prompt import (
    DEFAULT_TEXT_QA_PROMPT,
    DEFAULT_TREE_SUMMARIZE_PROMPT,
)

DEFAULT_TEXT_QA_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_TEXT_QA_PROMPT,
    conditionals=[(is_chat_model, CHAT_TEXT_QA_PROMPT)],
)

DEFAULT_TREE_SUMMARIZE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=DEFAULT_TREE_SUMMARIZE_PROMPT,
    conditionals=[(is_chat_model, CHAT_TREE_SUMMARIZE_PROMPT)],
)