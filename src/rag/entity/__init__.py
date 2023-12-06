from dataclasses import dataclass
from typing import Dict, Any, Union

from .schema import *
from .base_reader import *
from .base_vector import *

@dataclass
class MilvusConfig:
    host: str
    port: str
    address: str
    uri: str
    user: str

@dataclass
class MilvusArguments:
    collection_name: str
    consistency_level: str
    index_params: Dict[str, Any]
    search_params: Dict[str, Any]
    overwrite: False
    primary_field: str
    text_field: str
    embedding_field: str
    embedding_dim: int
