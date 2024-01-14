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

@dataclass
class LlmParams:
    context_window: int
    max_new_tokens: int
    model_name: str
    tokenizer_name: str
    device_map: str
    top_p: int
    top_k: int
    temperature: float
    length_penalty: float
    repetition_penalty: float
    num_beams: int
    do_sample : bool
    pelnaty_alpha: float
    use_cache: bool
    num_return_sequences: int
    pad_token_id: str
    bos_token_id: str
    eos_token_id: str

@dataclass
class EmbeddingsParams:
    model_name: str
    tokenizer_name: str
    pooling: str
    max_length: int
    normalize: bool
    embedding_batch_size: int
    cache_folder: str
    trust_remote_code: bool

@dataclass
class IndexRetrieveParams:
    index_type: str
    similarity_top_k: int
    alpha: float
    sparse_top_k: int 
    # list
    list_query_mode: str
    #keyword table
    keyword_table_model: str
    max_keywords_per_chunk: int
    max_keywords_per_query: int
    num_chunks_per_query: int
    # for vector store
    vector_store_query_mode: str
    store_nodes_override: bool
    insert_batch_size: int
    use_async: bool
    show_progress: bool

@dataclass
class ResponseParams:
    verbose: bool
    response_mode: str
    use_async: bool
    streaming: bool