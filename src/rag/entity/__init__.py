from dataclasses import dataclass
from typing import Dict, Any

from .callbacks import *
from .embeddings import *
from .indices import *
from .llm import *
from .node import *
from .node_parser import *
from .prompt import *
from .reader import *
from .retriever import *
from .storage import *
from .synthesizer import *
from .vector_store import *

from .schema import *
from .base_query_engine import *
from .storage_context import *
from .service_context import *
from .output_parser import *

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
