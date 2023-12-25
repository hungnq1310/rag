"""Base schema for data structures."""
import json
from abc import abstractmethod
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Generic,
    List,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
    Dict
)

from rag.bridge.pydantic import BaseModel
from rag.entity.llm.base_llm import ChatMessage, MessageRole
from typing_extensions import Self


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


Model = TypeVar("Model", bound=BaseModel)

TokenGen = Generator[str, None, None]
TokenAsyncGen = AsyncGenerator[str, None]
RESPONSE_TEXT_TYPE = Union[BaseModel, str, TokenGen]


# TODO: move into a `core` folder
# NOTE: this is necessary to make it compatible with pydantic
@runtime_checkable
class BaseOutputParser(Protocol):
    """Output parser class."""

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""

    @abstractmethod
    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""

    def format_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Format a list of messages with structured output formatting instructions."""
        # NOTE: apply output parser to either the first message if it's a system message
        #       or the last message
        if messages:
            if messages[0].role == MessageRole.SYSTEM:
                messages[0].content = self.format(messages[0].content or "")
            else:
                messages[-1].content = self.format(messages[-1].content or "")

        return messages


class BasePydanticProgram(ABC, Generic[Model]):
    """A base class for LLM-powered function that return a pydantic model.

    Note: this interface is not yet stable.
    """

    @property
    @abstractmethod
    def output_cls(self) -> Type[Model]:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Model:
        pass

    async def acall(self, *args: Any, **kwds: Any) -> Model:
        return self(*args, **kwds)


class PydanticProgramMode(str, Enum):
    """Pydantic program mode."""

    DEFAULT = "default"
    OPENAI = "openai"
    LLM = "llm"
    GUIDANCE = "guidance"
    LM_FORMAT_ENFORCER = "lm-format-enforcer"
