import logging
from typing import Optional, List

from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers.base import BaseSynthesizer
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index import (
    VectorStoreIndex,
    StorageContext,
    QueryBundle
)

from rag.entity.service_context import ServiceContext
from rag.components.retriever_engine import RetrieverQueryEngine
from rag.components.vector_store.milvus import MilvusVectorStore
from rag.components.subquestion_engine import SubQuestionQueryEngine
from rag.components.prompt_template import PromptTemplate
from rag.components.question_generator import LLMQuestionGenerator

logger = logging.getLogger(__name__)

class SubquetionPipeline:
    def __init__(self,
        config,
        params,
        prompt: PromptTemplate,
        service_context: Optional[ServiceContext],
        response_synthesizer: Optional[BaseSynthesizer],
        node_postprocessors: Optional[List[BaseNodePostprocessor]],
    ) -> None:
        self.config = config
        self.params = params
        self.prompt = prompt
        self.service_context = service_context
        self.response_synthesizer = response_synthesizer
        self.node_postprocessors = node_postprocessors       

    def run(self, query, documents):
        
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
        #TODO: call subquestion or from defaults
        subquestion = SubQuestionQueryEngine(
            question_gen=question_generator,
            response_synthesizer=self.response_synthesizer,
            query_engine_tools=[retriever_engine],
            verbose=self.params.verbose,
            use_async=self.params.use_async
        )
        #TODO: get response
        response = subquestion.query(QueryBundle(query))
        return response
