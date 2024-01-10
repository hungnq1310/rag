"""Base schema for data structures."""
import json
from abc import abstractmethod
from typing import (
    Any,
    List,
    Dict
)

from rag.bridge.pydantic import BaseModel
from typing_extensions import Self
from .node import BaseNode

class BaseComponent(BaseModel):
    """Base component object to capture class names."""

    class Config:
        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: "BaseComponent") -> None:
            """Add class name to schema."""
            schema["properties"]["class_name"] = {
                "title": "Class Name",
                "type": "string",
                "default": model.class_name(),
            }

    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name, used as a unique ID in serialization.

        This provides a key that makes serialization robust against actual class
        name changes.
        """
        return "base_component"

    def json(self, **kwargs: Any) -> str:
        return self.to_json(**kwargs)

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = super().dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()

        # tiktoken is not pickleable
        state["__dict__"].pop("tokenizer", None)

        # remove local functions
        keys_to_remove = []
        for key in state["__dict__"]:
            if key.endswith("_fn"):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            state["__dict__"].pop(key, None)

        # remove private attributes
        state["__private_attribute_values__"] = {}

        return state

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    # TODO: return type here not supported by current mypy version
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignore
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any) -> Self:  # type: ignore
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)


class TransformComponent(BaseComponent):
    """Base class for transform components."""

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def __call__(self, nodes: List["BaseNode"], **kwargs: Any) -> List["BaseNode"]:
        """Transform nodes."""

    async def acall(self, nodes: List["BaseNode"], **kwargs: Any) -> List["BaseNode"]:
        """Async transform nodes."""
        return self.__call__(nodes, **kwargs)
