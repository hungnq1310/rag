from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from rag.bridge.pydantic import BaseModel, Field
from rag.entity.llm import BaseLLM, ChatMessage
from rag.entity.output_parser import BaseOutputParser

class BasePromptTemplate(BaseModel, ABC):
    metadata: Dict[str, Any]
    template_vars: List[str]
    kwargs: Dict[str, str]
    output_parser: Optional[BaseOutputParser]
    template_var_mappings: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Template variable mappings (Optional)."
    )
    function_mappings: Optional[Dict[str, Callable]] = Field(
        default_factory=dict,
        description=(
            "Function mappings (Optional). This is a mapping from template "
            "variable names to functions that take in the current kwargs and "
            "return a string."
        ),
    )

    def _map_template_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """For keys in template_var_mappings, swap in the right keys."""
        template_var_mappings = self.template_var_mappings or {}
        return {template_var_mappings.get(k, k): v for k, v in kwargs.items()}

    def _map_function_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """For keys in function_mappings, compute values and combine w/ kwargs.

        Users can pass in functions instead of fixed values as format variables.
        For each function, we call the function with the current kwargs,
        get back the value, and then use that value in the template
        for the corresponding format variable.

        """
        function_mappings = self.function_mappings or {}
        # first generate the values for the functions
        new_kwargs = {}
        for k, v in function_mappings.items():
            # TODO: figure out what variables to pass into each function
            # is it the kwargs specified during query time? just the fixed kwargs?
            # all kwargs?
            new_kwargs[k] = v(**kwargs)

        # then, add the fixed variables only if not in new_kwargs already
        # (implying that function mapping will override fixed variables)
        for k, v in kwargs.items():
            if k not in new_kwargs:
                new_kwargs[k] = v

        return new_kwargs

    def _map_all_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Map both template and function variables.

        We (1) first call function mappings to compute functions,
        and then (2) call the template_var_mappings.

        """
        # map function
        new_kwargs = self._map_function_vars(kwargs)
        # map template vars (to point to existing format vars in string template)
        return self._map_template_vars(new_kwargs)

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def partial_format(self, **kwargs: Any) -> "BasePromptTemplate":
        ...

    @abstractmethod
    def format(self, llm: Optional[BaseLLM] = None, **kwargs: Any) -> str:
        ...

    @abstractmethod
    def format_messages(
        self, llm: Optional[BaseLLM] = None, **kwargs: Any
    ) -> List[ChatMessage]:
        ...

    @abstractmethod
    def get_template(self, llm: Optional[BaseLLM] = None) -> str:
        ...
