from omegaconf import DictConfig
from typing import Union

from .schema import (
    MilvusConfig,
    NodeParserConfig,
    LlmConfig,
    EmbeddingConfig,
    RetrieverConfig,
    ResponseConfig,
    FaissConfig,
    RerankConfig
)

class ConfigurationManager: 
    def __init__(
        self, 
        config: DictConfig, 
    ) -> None:
        self.config = config
        print(self.config)

    def get_db_config(self) -> Union[MilvusConfig, FaissConfig, ]: 
        """create instace for data ingestion config"""
        db_config = self.config.db
        if db_config.vectorstore_name == "milvus":
            return MilvusConfig(
                **db_config
            )
        elif db_config.vectorstore_name == "faiss":
            return FaissConfig(
                **db_config
            )
        else:
            raise ValueError("Not supported vector store. Please use either milvus or faiss.")

    def get_splitter_config(self) -> NodeParserConfig: 
        configs = self.config.splitter
        return NodeParserConfig(
            **configs
        )

    def get_llm_config(self) -> LlmConfig:
        """Currently support local llm model only."""
        configs = self.config.llm
        llm_config = LlmConfig(
            **configs
        )
        return llm_config
    
    def get_embed_config(self) -> EmbeddingConfig:
        configs = self.config.embedding
        return EmbeddingConfig(
            **configs
        )

    def get_index_retriever_config(self) -> RetrieverConfig:
        configs = self.config.retriever
        return RetrieverConfig(
            **configs
        )
    
    def get_cohere_rerank_config(self) -> RerankConfig:
        configs = self.config.rerank_config
        return RerankConfig(
            **configs
        )
    
    def get_response_config(self) -> ResponseConfig:
        return ResponseConfig(
            verbose=self.config.verbose,
            response_mode=self.config.response_mode,
            use_async=self.config.use_async,
            streaming=self.config.streaming,
        )
