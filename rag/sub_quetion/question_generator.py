from typing import List, Optional, Sequence, cast




from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.output_parsers.base import StructuredOutput
from llama_index.question_gen.prompts import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL,
    build_tools_text,
)
from llama_index.tools.types import ToolMetadata

from rag.prompt import PromptType
from rag.retriever.base_retriver import QueryBundle
from rag.core.service_context import ServiceContext
from rag.schema.component import BaseOutputParser
from rag.prompt.base_prompt import BasePromptTemplate
from rag.sub_quetion import BaseQuestionGenerator, SubQuestion
from rag.sub_quetion.output_parser import SubQuestionOutputParser
from rag.prompt.mixin import PromptDictType
from rag.prompt.prompt_template import PromptTemplate



class LLMQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        llm_predictor: BaseLLMPredictor,
        prompt: BasePromptTemplate,
    ) -> None:
        self._llm_predictor = llm_predictor
        self._prompt = prompt

        if self._prompt.output_parser is None:
            raise ValueError("Prompt should have output parser.")

    @classmethod
    def from_defaults(
        cls,
        service_context: Optional[ServiceContext] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> "LLMQuestionGenerator":
        # optionally initialize defaults
        service_context = service_context or ServiceContext.from_defaults()
        prompt_template_str = prompt_template_str or DEFAULT_SUB_QUESTION_PROMPT_TMPL
        output_parser = output_parser or SubQuestionOutputParser()

        # construct prompt
        prompt = PromptTemplate(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.SUB_QUESTION,
        )
        return cls(service_context.llm_predictor, prompt)

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"question_gen_prompt": self._prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "question_gen_prompt" in prompts:
            self._prompt = prompts["question_gen_prompt"]

    def generate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = self._llm_predictor.predict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output

    async def agenerate(
        self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = await self._llm_predictor.apredict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        return parse.parsed_output