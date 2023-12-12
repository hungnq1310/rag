import logging 
from typing import Optional, List

from llama_index import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.retrievers import VectorIndexRetriever

from rag.components.retriever_engine import RetrieverQueryEngine
from rag.components.milvus_vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)

class RetrieverPipeline:
    def __init__(self,
        config,
        params ,
        service_context: Optional[ServiceContext],
        response_synthesizer: Optional[BaseSynthesizer],
        node_postprocessors: Optional[List[BaseNodePostprocessor]],
    ) -> None:
        self.config = config
        self.params = params
        self.service_context = service_context
        self.response_synthesizer = response_synthesizer
        self.node_postprocessors = node_postprocessors                    

    def main(self, query, documents):
        #TODO: build milvus vector from documents
        milvus_vector_store = MilvusVectorStore(self.config, self.params)
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
            similarity_top_k=self.params.similarity_top_k,
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

