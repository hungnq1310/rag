from typing import List, Optional, Sequence, cast, TYPE_CHECKING

from rag.prompt.types import PromptType

from rag.prompt.prompt_template import PromptTemplate
from rag.output_parser.types import StructuredOutput
from .base import BaseQuestionGenerator, SubQuestion
from .prompt_gen import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL,
    build_tools_text,
)

if TYPE_CHECKING:
    from rag.llm.base import LLM
    from rag.tools.types import ToolMetadata
    from rag.core.service_context import ServiceContext
    from rag.retrievers.base_retriver import QueryBundle
    from rag.prompt.base_prompt import BasePromptTemplate
    from rag.output_parser.base import BaseOutputParser
    from rag.prompt.mixin import PromptDictType


class LLMQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        llm: "LLM",
        prompt: "BasePromptTemplate",
    ) -> None:
        self._llm = llm
        self._prompt = prompt

        if self._prompt.output_parser is None:
            raise ValueError("Prompt should have output parser.")

    @classmethod
    def from_defaults(
        cls,
        service_context: "ServiceContext" = None,
        prompt_template_str: Optional[str] = None,
        output_parser: "BaseOutputParser" = None,
    ) -> "LLMQuestionGenerator":
        # optionally initialize defaults
        service_context = service_context
        prompt_template_str = prompt_template_str or DEFAULT_SUB_QUESTION_PROMPT_TMPL
        output_parser = output_parser

        # construct prompt
        prompt = PromptTemplate(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.SUB_QUESTION,
        )
        return cls(service_context.llm, prompt)

    def _get_prompts(self) -> "PromptDictType":
        """Get prompts."""
        return {"question_gen_prompt": self._prompt}

    def _update_prompts(self, prompts: "PromptDictType") -> None:
        """Update prompts."""
        if "question_gen_prompt" in prompts:
            self._prompt = prompts["question_gen_prompt"]

    def generate(
        self, tools: Sequence["ToolMetadata"], query: "QueryBundle"
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = self._llm.predict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output

    async def agenerate(
        self, tools: Sequence["ToolMetadata"], query: "QueryBundle"
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = await self._llm.apredict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output