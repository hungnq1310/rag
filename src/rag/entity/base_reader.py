"""Base reader class."""
from abc import ABC
from typing import Any, Iterable, List

from rag.entity.schema import Document

class BaseReader(ABC):
    """Utilities for loading data from a directory."""

    def lazy_load_data(self, *args: Any, **load_kwargs: Any) -> Iterable[Document]:
        """Load data from the input directory lazily."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide lazy_load_data method currently"
        )

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        return list(self.lazy_load_data(*args, **load_kwargs))
