from typing import Any, Sequence, TYPE_CHECKING

from rag.entity.synthesizer import BaseSynthesizer

if TYPE_CHECKING:
    from rag.entity.output_parser import RESPONSE_TEXT_TYPE
    from rag.entity.prompt.mixin import PromptDictType


class NoText(BaseSynthesizer):
    def _get_prompts(self) -> "PromptDictType":
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: "PromptDictType") -> None:
        """Update prompts."""

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> "RESPONSE_TEXT_TYPE":
        return ""

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> "RESPONSE_TEXT_TYPE":
        return ""