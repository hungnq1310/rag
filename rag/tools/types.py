from typing import Optional, Type, Dict, Any
import json
from pydantic import BaseModel
from dataclasses import dataclass


class DefaultToolFnSchema(BaseModel):
    """Default tool function Schema."""

    input: str


@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None
    fn_schema: Optional[Type[BaseModel]] = DefaultToolFnSchema

    def get_parameters_dict(self) -> dict:
        if self.fn_schema is None:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {"title": "input query string", "type": "string"},
                },
                "required": ["input"],
            }
        else:
            parameters = self.fn_schema.schema()
            parameters = {
                k: v
                for k, v in parameters.items()
                if k in ["type", "properties", "required", "definitions"]
            }
        return parameters

    @property
    def fn_schema_str(self) -> str:
        """Get fn schema as string."""
        if self.fn_schema is None:
            raise ValueError("fn_schema is None.")
        parameters = self.get_parameters_dict()
        return json.dumps(parameters)

    def get_name(self) -> str:
        """Get name."""
        if self.name is None:
            raise ValueError("name is None.")
        return self.name


class ToolOutput(BaseModel):
    """Tool output."""

    content: str
    tool_name: str
    raw_input: Dict[str, Any]
    raw_output: Any

    def __str__(self) -> str:
        """String."""
        return str(self.content)