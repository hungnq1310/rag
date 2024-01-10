from abc import abstractmethod
from typing import Any, Sequence

from llama_index.bridge.pydantic import Field, validator
from rag.entity.callbacks import CallbackManager
from rag.entity.schema import BaseComponent
from rag.entity.prompt import BasePromptTemplate
from rag.entity.output_parser import TokenAsyncGen, TokenGen
from .types import *

class BaseLLM(BaseComponent):
    """LLM interface."""

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    @validator("callback_manager", pre=True)
    def _validate_callback_manager(cls, v: CallbackManager) -> CallbackManager:
        if v is None:
            return CallbackManager([])
        return v

    @property
    @abstractmethod
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint for LLM."""

    @abstractmethod
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Streaming chat endpoint for LLM."""

    @abstractmethod
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Streaming completion endpoint for LLM."""

    # ===== Async Function =====
    @abstractmethod
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat endpoint for LLM."""

    @abstractmethod
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async completion endpoint for LLM."""

    @abstractmethod
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint for LLM."""

    @abstractmethod
    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion endpoint for LLM."""

    @abstractmethod
    def predict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> str:
        """Calling complete process from prompt to response"""

    @abstractmethod
    def stream(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> TokenGen:
        """Streaming complete process from prompt to response"""


    @abstractmethod
    async def apredict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> str:
        """Async calling complete process from prompt to response"""


    @abstractmethod
    async def astream(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> TokenAsyncGen:
        """Async calling streaming process from prompt to response"""




