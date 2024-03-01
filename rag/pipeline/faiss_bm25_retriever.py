import logging 
import time
import faiss
from typing import List
from transformers import AutoTokenizer

from rag.config.schema import (
    SpiltterConfig,
    EmbeddingConfig,
    IndexRetrieverConfig,
    ResponseConfig,
    FaissConfig,
    CohereRerankConfig
)
from rag.node_parser import SentenceSplitter
from rag.callbacks import CallbackManager
from rag.core.storage_context import StorageContext
from rag.core.service_context import ServiceContext
from rag.indices.vector_store import VectorStoreIndex
from rag.vector_stores.faiss import FaissVectorStore 

from rag.retrievers.hybrid_retriever import HybridSearchRetriever
from rag.retrievers.dense.vector_retriver import VectorIndexRetriever
from rag.retrievers.sparse.bm25 import BM25Retriever
from rag.retrievers.types import QueryBundle, QueryType

from rag.engine.retriever_engine import RetrieverQueryEngine
from rag.core.prompt_helper import PromptHelper
from rag.synthesizer.utils import get_response_synthesizer
from rag.node.base_node import Document
from rag.rerank.cohere_rerank import CohereRerank
from rag.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

class FaissRetrieverPipeline:
    def __init__(
        self,
        documents: List[Document],
        splitter_config: SpiltterConfig,
        index_retriver_config: IndexRetrieverConfig,
        embed_config: EmbeddingConfig,
        faiss_config: FaissConfig,
        rerank_config: CohereRerankConfig,
        response_config: ResponseConfig,
    ) -> None:
        self.splitter_config = splitter_config
        self.index_retriver_config = index_retriver_config
        self.faiss_config = faiss_config
        self.rerank_config = rerank_config
        self.response_config = response_config

        #callback manager
        callback_manager = CallbackManager()

        # node parser
        tokenizer = AutoTokenizer.from_pretrained(self.splitter_config.model_name_tokenizer)
        node_parser = SentenceSplitter(
            separator= self.splitter_config.separator,
            chunk_size= self.splitter_config.chunk_size,
            chunk_overlap= self.splitter_config.chunk_overlap,
            tokenizer= tokenizer.encode,
            paragraph_separator= self.splitter_config.paragraph_separator,
            secondary_chunking_regex= self.splitter_config.secondary_chunking_regex,
            callback_manager= callback_manager,
        )

        #embed model
        embed_model = self.get_embed_mode(embed_config)


        # check if embed_model.embedding_dim == faiss_config.embedding_dim
        assert embed_model.get_model_dim() == self.faiss_config.embedding_dim, f"""
        The embedding_dim of embed_model: {embed_model.get_model_dim()} is not equal to the embedding_dim of faiss_config: {self.faiss_config.embedding_dim}"""

        # resole index_type for faiss
        if self.faiss_config.index_build == "l2":
            self.faiss_index = faiss.IndexFlatL2(self.faiss_config.embedding_dim)
        elif self.faiss_config.index_build == "ip":
            self.faiss_index = faiss.IndexFlatIP(self.faiss_config.embedding_dim)
        elif self.faiss_config.index_build == "ivf":
            self.faiss_index = faiss.IndexIVFFlat(self.faiss_config.embedding_dim)


        # prompt helper - tong hop 3 cai params cua llm, emb, node parser
        prompt_helper = PromptHelper()

        self.service_context = ServiceContext(
            # llm=llm,
            prompt_helper=prompt_helper,
            embed_model=embed_model,
            node_parser=node_parser,
            callback_manager=callback_manager,
        )

        #TODO: build faiss vector from documents
        faiss_vector = FaissVectorStore(
            faiss_index= self.faiss_index,
        )

        # construct index and customize storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store= faiss_vector
        )

        nodes = []

        for each_parser in self.service_context.transformations: 
            parsing_nodes = each_parser.get_nodes_from_documents(
                documents=documents,
                show_progress=True,
            )
            nodes.extend(parsing_nodes)


        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context= self.storage_context,
            service_context= self.service_context,
            store_nodes_override= self.index_retriver_config.store_nodes_override,
            insert_batch_size= self.index_retriver_config.insert_batch_size,
            use_async= self.index_retriver_config.use_async,
            show_progress= self.index_retriver_config.show_progress,
        )

        #TODO: build retrievers
        self.dense_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k= self.index_retriver_config.similarity_top_k,
            sparse_top_k= self.index_retriver_config.sparse_top_k,
            alpha= self.index_retriver_config.alpha,
            vector_store_query_mode= self.index_retriver_config.vector_store_query_mode,
        )
        
        self.sparse_retriver = BM25Retriever(
            nodes= nodes,
            similarity_top_k= self.index_retriver_config.sparse_top_k,
            callback_manager= self.service_context.callback_manager,
        )
              

    def main(self, query: QueryType):

        """Wrap time"""
        start_build_collection_index = int(round(time.time() * 1000))

        end_build_collection_index = int(round(time.time() * 1000))
        print(f"Time for build collection and index: {end_build_collection_index - start_build_collection_index} ms")

        """Wrap for loading retriever and search"""
        start_build_retrieve_search = int(round(time.time() * 1000))

        # build Hybrid Search Retriever
        retriever = HybridSearchRetriever(
            sparse_retriever=self.sparse_retriver,
            dense_retriever=self.dense_retriever,
            mode=self.index_retriver_config.hybrid_mode,
            similarity_top_k=self.index_retriver_config.similarity_top_k,
            callback_manager=self.service_context.callback_manager,
        )

        # response_synthesizer
        response_synthesizer = get_response_synthesizer(
            service_context= self.service_context,
            callback_manager= self.service_context.callback_manager,
            response_mode= self.response_config.response_mode,
            verbose= self.response_config.verbose,
            use_async= self.response_config.use_async,
            streaming= self.response_config.streaming,
        ) 

        # cohere rerank
        node_processor = CohereRerank(
            top_n= self.rerank_config.top_n,
            model= self.rerank_config.model,
            api_key= self.rerank_config.api_key,
        )

        #TODO: assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever= retriever,
            response_synthesizer= response_synthesizer,
            node_postprocessors= [node_processor]
        )
        #TODO: query
        if isinstance(query, str):
            query = QueryBundle(query_str= query)
        nodes = query_engine.retrieve(query)

        end_build_retrieve_search = int(round(time.time() * 1000))
        print(f"Time for build retriever and search: {end_build_retrieve_search - start_build_retrieve_search} ms")

        return nodes

    def get_embed_mode(self, config: EmbeddingConfig) -> HuggingFaceEmbedding:
        # emb_model
        emb_model = HuggingFaceEmbedding(
            model_name= config.model_name,
            tokenizer_name= config.tokenizer_name,
            pooling= config.pooling,
            max_length= config.max_length,
            normalize= config.normalize,
            embed_batch_size= config.embedding_batch_size,
            cache_folder= config.cache_folder,
            trust_remote_code= config.trust_remote_code,
        )
        return emb_model
