from typing import (
    Dict,
    Any,
    List
)
from dataclasses import dataclass

@dataclass
class MilvusConfig:
    vectorstore_name: str
    collection_name: str
    insert_batch_size: int
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
    overwrite: bool
    index_params: Dict[str, Any]
    search_params: Dict[str, Any]

@dataclass
class FaissConfig:
    vectorstore_name: str
    embedding_dim: int
    index_build: str

@dataclass
class NodeParserConfig:
  model_name_tokenizer: str
  splitter_mode: str
  separator: str
  chunk_size: int
  chunk_overlap: int
  # for sentence splitter 
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
    token: str
    device: str
    cache_folder: str
    trust_remote_code: bool
    use_async: bool
    show_progress: bool

@dataclass
class RetrieverConfig:
    retriever_mode: str
    similarity_top_k: int
    use_async: bool
    show_progress: bool
    # list
    list_query_mode: str
    choice_batch_size: int
    #keyword table
    keyword_table_mode: str
    max_keywords_per_chunk: int
    max_keywords_per_query: int
    num_chunks_per_query: int
    # for vector store
    vector_store_query_mode: str
    alpha: float
    sparse_top_k: int 

@dataclass
class RerankConfig:
    modes: List[str]
    use_async: bool
    show_progress: bool
    # cohere
    top_n: int
    model: str
    api_key: str
    # sbert
    model_name: str
    token: str
    device: str
    # huggingface
    tokenizer_name: str
    max_length: int
    keep_retrieval_score: bool
    # llm
    choice_batch_size: int
    # simple
    delta_similarity_cutoff: float

@dataclass
class ResponseConfig:
    verbose: bool
    response_mode: str
    use_async: bool
    streaming: bool