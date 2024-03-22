from typing import Any, Optional

from rag.question_gen.base import SubQuestion
from rag.question_gen.utils import parse_json_markdown

from .base import BaseOutputParser
from .types import StructuredOutput


class SubQuestionOutputParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        json_dict = parse_json_markdown(output)
        if not json_dict:
            raise ValueError(f"No valid JSON found in output: {output}")

        sub_questions = [SubQuestion.parse_obj(item) for item in json_dict]
        return StructuredOutput(raw_output=output, parsed_output=sub_questions)

    def format(self, prompt_template: str) -> str:
        return prompt_template
