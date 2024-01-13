from typing import (
    Dict,
    Any,
)
from dataclasses import dataclass

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

@dataclass
class NodeParserConfig:
    name: str


@dataclass
class NodeParserArguments:
  model_name_tokenizer: str
  # fpr sentence splitter 
  splitter_mode: str
  separator: str
  chunk_size: int
  chunk_overlap: int
  paragraph_separator: str
  secondary_chunking_regex: str
  # for token splitter
  backup_separators: str