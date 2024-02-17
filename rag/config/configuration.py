from pathlib import Path
from rag.constants import *
from rag.rag_utils.files import (
    read_yaml, 
    create_directories
)
from .schema import (
    MilvusConfig,
    MilvusArguments,
    NodeParserArguments,
    NodeParserConfig,
    LlmParams,
    EmbeddingsParams,
    IndexRetrieveParams,
    ResponseParams
)

class ConfigurationManager: 
    def __init__(
                self, 
                config_filepath: Path = CONFIG_FILE_PATH, 
                param_filepath: Path = PARAMS_FILE_PATH
                ) -> None:
        self.config = read_yaml(config_filepath)
        self.param = read_yaml(param_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_milvus_config(self) -> MilvusConfig: 
        """create instace for data ingestion config"""
        configs = self.config.milvus_config

        milvus_config = MilvusConfig(
            host= configs.host,
            port= configs.port,
            address= configs.address,
            uri= configs.uri,
            user= configs.user,
        )
        return milvus_config
    
    def get_milvus_params(self) -> MilvusArguments:
        params = self.param.MilvusParams
        milvus_params = MilvusArguments(
            collection_name=params.collection_name,
            consistency_level=params.consistency_level,
            index_params=params.index_params,
            search_params=params.search_params,
            overwrite=params.overwrite,
            primary_field=params.primary_field,
            text_field=params.text_field,
            embedding_field=params.embedding_field,
            embedding_dim=params.embedding_dim,
        )
        return milvus_params
    
    def get_node_parser_config(self) -> NodeParserConfig: 
        config = self.config.node_parser_config
        node_parser_config = NodeParserConfig(
            name=config.name
        )
        return node_parser_config

    def get_node_parser_params(self) -> NodeParserArguments: 
        params = self.param.NodeParserParams
        node_parser_params = NodeParserArguments(
            model_name_tokenizer=params.model_name_tokenizer,
            splitter_mode=params.splitter_mode,
            separator=params.separator,
            chunk_size=params.chunk_size,
            chunk_overlap=params.chunk_overlap,
            paragraph_separator=params.paragraph_separator,
            secondary_chunking_regex=params.secondary_chunking_regex,
            backup_separators=params.backup_separators,
        )
        return node_parser_params
            

    def get_llm_params(self) -> LlmParams:
        params = self.param.LlmParams
        llm_params = LlmParams(
            context_window=params.context_window,
            max_new_tokens=params.max_new_tokens,
            model_name=params.model_name,
            tokenizer_name=params.tokenizer_name,
            device_map=params.device_map,
            top_k=params.top_k,
            top_p=params.top_p,
            temperature=params.temperature,
            length_penalty=params.length_penalty,
            repetition_penalty=params.repetition_penalty,
            num_beams=params.num_beams,
            do_sample=params.do_sample,
            pelnaty_alpha=params.pelnaty_alpha,
            use_cache=params.use_cache,
            num_return_sequences=params.num_return_sequences,
            pad_token_id=params.pad_token_id,
            bos_token_id=params.bos_token_id,
            eos_token_id=params.eos_token_id,
        )
        return llm_params
    
    def get_embed_params(self) -> EmbeddingsParams:
        params = self.param.embeddings_params
        embed_params = EmbeddingsParams(
            model_name= params.model_name,
            tokenizer_name= params.tokenizer_name,
            pooling= params.pooling,
            max_length= params.max_length,
            normalize= params.normalize,
            embedding_batch_size= params.embedding_batch_size,
            cache_folder= params.cache_folder,
            trust_remote_code= params.trust_remote_code,
        )
        return embed_params

    def get_index_retriever_params(self) -> IndexRetrieveParams:
        params = self.param.IndexRetrieveParams
        embed_params = IndexRetrieveParams(
            index_type=params.index_type,
            similarity_top_k=params.similarity_top_k,
            sparse_top_k=params.sparse_top_k,
            alpha=params.alpha,
            list_query_mode=params.list_query_mode,
            keyword_table_mode=params.keyword_table_mode,
            max_keywords_per_chunk=params.max_keywords_per_chunk,
            max_keywords_per_query=params.max_keywords_per_query,
            num_chunks_per_query=params.num_chunks_per_query,
            vector_store_query_mode=params.vector_store_query_mode,
            store_nodes_override=params.store_nodes_override,
            insert_batch_size=params.insert_batch_size,
            use_async=params.use_async,
            show_progress=params.show_progress,
        )
        return embed_params
    
    def get_response_params(self) -> ResponseParams:
        params = self.param.ResponseParams
        response_params = ResponseParams(
            verbose=params.verbose,
            response_mode=params.response_mode,
            use_async=params.use_async,
            streaming=params.streaming,
        )
        return response_params