import logging
from typing import Optional, List, Any

from llama_index import VectorStoreIndex, StorageContext, QueryBundle
from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.postprocessor.types import BaseNodePostprocessor

from rag.entity.service_context import ServiceContext
from rag.components.question_generator import LLMQuestionGenerator
from rag.components.engine.retriever_engine import RetrieverQueryEngine
from rag.components.vector_store.milvus import MilvusVectorStore

logger = logging.getLogger(__name__)

class RewritePipeline:
    def __init__(
        self,
        config,
        params,
        service_context: Optional[ServiceContext],
        response_synthesizer: Optional[BaseSynthesizer],
        node_postprocessors: Optional[List[BaseNodePostprocessor]],
    ) -> None:
        self.config = config
        self.params = params
        self.service_context = service_context
        self.response_synthesizer = response_synthesizer
        self.node_postprocessors = node_postprocessors       

    def run(self, query: QueryBundle, documents: List[Any]):
        
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

        # build retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.params.similarity_top_k,
            vector_store_query_mode=self.params.vector_store_query_mode,
            alpha=self.params.alpha,
            doc_ids=None,
        )

        #TODO: call question generator from default
        question_generator = LLMQuestionGenerator(
            llm_predictor=self.service_context.llm_predictor,
            prompt=self.prompt
        )
        #TODO: call retriever or from defaults
        retriever_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=self.response_synthesizer,
            node_postprocessors=self.node_postprocessors
        )

        #TODO: init rewrite engine
        """
        (1) gen and rewrite sub question
        (2) retrieve with sub question
        (3) select top k from subquestion.sources
        (4) gen response with top-k docs
        (5) return  
        """
        rewrite_engine = None
        
        #TODO: return response   
        respones = None
        return respones