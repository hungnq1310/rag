from typing import Any, Optional, Sequence

from rag.constants.default_prompt import DEFAULT_SIMPLE_INPUT_PROMPT
from rag.components.prompt.mixin import PromptDictType
from rag.entity.prompt import BasePromptTemplate
from rag.entity.synthesizer import BaseSynthesizer
from rag.entity.service_context import ServiceContext
from rag.entity.schema import RESPONSE_TEXT_TYPE


class Generation(BaseSynthesizer):
    def __init__(
        self,
        simple_template: Optional[BasePromptTemplate] = None,
        service_context: Optional[ServiceContext] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(service_context=service_context, streaming=streaming)
        self._input_prompt = simple_template or DEFAULT_SIMPLE_INPUT_PROMPT

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"simple_template": self._input_prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "simple_template" in prompts:
            self._input_prompt = prompts["simple_template"]

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        # NOTE: ignore text chunks and previous response
        del text_chunks

        if not self._streaming:
            return await self._service_context.llm.apredict(
                self._input_prompt,
                query_str=query_str,
                **response_kwargs,
            )
        else:
            return self._service_context.llm.stream(
                self._input_prompt,
                query_str=query_str,
                **response_kwargs,
            )

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        # NOTE: ignore text chunks and previous response
        del text_chunks

        if not self._streaming:
            return self._service_context.llm.predict(
                self._input_prompt,
                query_str=query_str,
                **response_kwargs,
            )
        else:
            return self._service_context.llm.stream(
                self._input_prompt,
                query_str=query_str,
                **response_kwargs,
            )