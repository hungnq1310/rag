from pathlib import Path
from rag.constants import *
from rag.rag_utils.files import (
    read_yaml, 
    create_directories
)
from .schema import (
    MilvusConfig,
    SpiltterConfig,
    LlmConfig,
    EmbeddingConfig,
    IndexRetrieverConfig,
    ResponseConfig,
    FaissConfig
)

class ConfigurationManager: 
    def __init__(
        self, 
        config_filepath: Path = CONFIG_FILE_PATH, 
    ) -> None:
        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])
    
    def get_milvus_config(self) -> MilvusConfig: 
        """create instace for data ingestion config"""
        configs = self.config.vector_store_config

        milvus_config = MilvusConfig(
            vectorstore_name=configs.vector_name,
            host= configs.host,
            port= configs.port,
            address= configs.address,
            uri= configs.uri,
            user= configs.user,
            collection_name=configs.collection_name,
            consistency_level=configs.consistency_level,
            index_params=configs.index_params,
            search_params=configs.search_params,
            overwrite=configs.overwrite,
            primary_field=configs.primary_field,
            text_field=configs.text_field,
            embedding_field=configs.embedding_field,
            embedding_dim=configs.embedding_dim,
        )
        return milvus_config
    
    
    def get_faiss_config(self) -> FaissConfig:
        config = self.config.faiss_config

        faiss_config = FaissConfig(
            vectorstore_name=config.vector_name,
            embedding_dim=config.embedding_dim,
            index_build=config.index_build,
        )
        return faiss_config

    
    def get_splitter_config(self) -> SpiltterConfig: 
        configs = self.config.splitter_config
        splitter_config = SpiltterConfig(
            model_name_tokenizer=configs.model_name_tokenizer,
            splitter_mode=configs.splitter_mode,
            separator=configs.separator,
            chunk_size=configs.chunk_size,
            chunk_overlap=configs.chunk_overlap,
            paragraph_separator=configs.paragraph_separator,
            secondary_chunking_regex=configs.secondary_chunking_regex,
            backup_separators=configs.backup_separators,
        )
        return splitter_config


    def get_llm_config(self) -> LlmConfig:
        config = self.config.llm_config
        llm_config = LlmConfig(
            context_window=config.context_window,
            max_new_tokens=config.max_new_tokens,
            model_name=config.model_name,
            tokenizer_name=config.tokenizer_name,
            device_map=config.device_map,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            length_penalty=config.length_penalty,
            repetition_penalty=config.repetition_penalty,
            num_beams=config.num_beams,
            do_sample=config.do_sample,
            pelnaty_alpha=config.pelnaty_alpha,
            use_cache=config.use_cache,
            num_return_sequences=config.num_return_sequences,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )
        return llm_config
    
    def get_embed_config(self) -> EmbeddingConfig:
        config = self.config.embeddings_config
        embed_config = EmbeddingConfig(
            model_name= config.model_name,
            tokenizer_name= config.tokenizer_name,
            pooling= config.pooling,
            max_length= config.max_length,
            normalize= config.normalize,
            embedding_batch_size= config.embedding_batch_size,
            cache_folder= config.cache_folder,
            trust_remote_code= config.trust_remote_code,
        )
        return embed_config

    def get_index_retriever_config(self) -> IndexRetrieverConfig:
        config = self.config.index_retriever_config
        embed_config = IndexRetrieverConfig(
            index_type=config.index_type,
            similarity_top_k=config.similarity_top_k,
            sparse_top_k=config.sparse_top_k,
            alpha=config.alpha,
            list_query_mode=config.list_query_mode,
            keyword_table_mode=config.keyword_table_mode,
            max_keywords_per_chunk=config.max_keywords_per_chunk,
            max_keywords_per_query=config.max_keywords_per_query,
            num_chunks_per_query=config.num_chunks_per_query,
            vector_store_query_mode=config.vector_store_query_mode,
            store_nodes_override=config.store_nodes_override,
            insert_batch_size=config.insert_batch_size,
            use_async=config.use_async,
            show_progress=config.show_progress,
        )
        return embed_config
    
    def get_response_config(self) -> ResponseConfig:
        config = self.config.response_config
        response_config = ResponseConfig(
            verbose=config.verbose,
            response_mode=config.response_mode,
            use_async=config.use_async,
            streaming=config.streaming,
        )
        return response_config
    
    def get_extract_adobe_pdf(self):
        return self.config.extract_adobe_pdf