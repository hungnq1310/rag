from pathlib import Path
from rag.constants import *
from rag.utils.files import (
    read_yaml, 
    create_directories
)
from rag.entity import (
    MilvusConfig,
    MilvusArguments
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
