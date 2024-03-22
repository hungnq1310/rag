from typing import (
    Dict,
    Any,
)
from dataclasses import dataclass

@dataclass
class MilvusConfig:
    vectorstore_name: str
    embedding_dim: int
    host: str
    port: str
    address: str
    uri: str
    user: str
    embedding_field: str
    primary_field: str
    text_field: str
    consistency_level: str
    collection_name: str
    index_params: Dict[str, Any]
    search_params: Dict[str, Any]
    overwrite: bool

@dataclass
class FaissConfig:
    vectorstore_name: str
    embedding_dim: int
    index_build: str

@dataclass
class SpiltterConfig:
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
class LlmConfig:
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
class EmbeddingConfig:
    model_name: str
    tokenizer_name: str
    pooling: str
    max_length: int
    normalize: bool
    embedding_batch_size: int
    cache_folder: str
    trust_remote_code: bool

@dataclass
class IndexRetrieverConfig:
    hybrid_mode: str
    similarity_top_k: int
    alpha: float
    sparse_top_k: int 
    # list
    list_query_mode: str
    #keyword table
    keyword_table_mode: str
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
class CohereRerankConfig:
    top_n: int
    model: str
    api_key: str

@dataclass
class ResponseConfig:
    verbose: bool
    response_mode: str
    use_async: bool
    streaming: bool