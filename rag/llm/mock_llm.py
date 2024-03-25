from typing import Any, Callable, Optional, Sequence

from rag.callbacks.callback_manager import CallbackManager
from rag.output_parser.output_parser import BaseOutputParser
from rag.prompt.base_prompt import BasePromptTemplate

from .types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from .utils import (
    llm_completion_callback,
)
from .base import LLM


class MockLLM(LLM):
    max_tokens: Optional[int]

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
        query_wrapper_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        max_tokens = max_tokens
        super().__init__(
            callback_manager=callback_manager, # type: ignore
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt, # type: ignore
            completion_to_prompt=completion_to_prompt, # type: ignore
            output_parser=output_parser,   
            query_wrapper_prompt=query_wrapper_prompt,    
        )

    @classmethod
    def class_name(cls) -> str:
        return "MockLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(num_output=self.max_tokens or -1)

    def _generate_text(self, length: int) -> str:
        return " ".join(["text" for _ in range(length)])

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response_text = (
            self._generate_text(self.max_tokens) if self.max_tokens else prompt
        )

        return CompletionResponse(
            text=response_text,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def gen_prompt() -> CompletionResponseGen:
            for ch in prompt:
                yield CompletionResponse(
                    text=prompt,
                    delta=ch,
                )

        def gen_response(max_tokens: int) -> CompletionResponseGen:
            for i in range(max_tokens):
                response_text = self._generate_text(i)
                yield CompletionResponse(
                    text=response_text,
                    delta="text ",
                )

        return gen_response(self.max_tokens) if self.max_tokens else gen_prompt()