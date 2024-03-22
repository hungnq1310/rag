from typing import Any, Optional
from dataclasses import dataclass

class OutputParserException(Exception):
    pass

@dataclass
class StructuredOutput:
    """Structured output class."""

    raw_output: str
    parsed_output: Optional[Any] = None