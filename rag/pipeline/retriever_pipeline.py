import logging 
from typing import Optional, List

from rag.config.schema import (
    MilvusConfig,
    MilvusArguments,
    IndexRetrieveParams
)
from rag.entity import VectorStoreIndex, StorageContext, ServiceContext
from rag.entity.synthesizer import BaseSynthesizer
from rag.entity.indices.vector_store import VectorIndexRetriever
from rag.components.engine.retriever_engine import RetrieverQueryEngine
from rag.components.vector_store.milvus import MilvusVectorStore
from llama_index.postprocessor.types import BaseNodePostprocessor

logger = logging.getLogger(__name__)

class RetrieverPipeline:
    def __init__(self,
        milvus_config: MilvusConfig,
        milvus_params: MilvusArguments,
        index_params: IndexRetrieveParams,
        service_context: Optional[ServiceContext],
        response_synthesizer: Optional[BaseSynthesizer],
        node_postprocessors: Optional[List[BaseNodePostprocessor]],
    ) -> None:
        self.milvus_config = milvus_config
        self.milvus_params = milvus_params
        self.index_params = index_params
        self.service_context = service_context
        self.response_synthesizer = response_synthesizer
        self.node_postprocessors = node_postprocessors                    

    def main(self, query, documents):
        #TODO: build milvus vector from documents
        milvus_vector_store = MilvusVectorStore(self.milvus_config, self.milvus_params)
        # construct index and customize storage context
        storage_context = StorageContext.from_defaults(
            vector_store=milvus_vector_store
        )
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            service_context=self.service_context,
        )
        #TODO: build retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.index_params.similarity_top_k,
        )

        #TODO: assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=self.response_synthesizer,
            node_postprocessors=self.node_postprocessors
        )
        #TODO: query
        response = query_engine.query(query)
        print(response)
        return response

