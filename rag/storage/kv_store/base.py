from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import fsspec

DEFAULT_COLLECTION = "data"


class BaseKVStore(ABC):
    """Base key-value store."""

    @abstractmethod
    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        pass

    @abstractmethod
    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        pass

    @abstractmethod
    def put_all(
        self, kv_pairs: List[Tuple[str, dict]], collection: str = DEFAULT_COLLECTION
    ) -> None:
        pass

    @abstractmethod
    async def aput_all(
        self, kv_pairs: List[Tuple[str, dict]], collection: str = DEFAULT_COLLECTION
    ) -> None:
        pass

    @abstractmethod
    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        pass

    @abstractmethod
    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        pass

    @abstractmethod
    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        pass

    @abstractmethod
    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        pass

    @abstractmethod
    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        pass

    @abstractmethod
    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        pass


class BaseInMemoryKVStore(BaseKVStore):
    """Base in-memory key-value store."""

    @abstractmethod
    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_persist_path(cls, persist_path: str) -> "BaseInMemoryKVStore":
        """Create a BaseInMemoryKVStore from a persist directory."""