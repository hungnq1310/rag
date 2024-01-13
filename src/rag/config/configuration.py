from pathlib import Path
from rag.constants import *
from rag.utils.files import (
    read_yaml, 
    create_directories
)
from .schema import (
    MilvusConfig,
    MilvusArguments,
    NodeParserArguments,
    NodeParserConfig
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
        params = self.param.milvus_params
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
            