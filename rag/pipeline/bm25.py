import logging 
import time

from transformers import AutoTokenizer

from rag.config.schema import (
    NodeParserConfig,
    RetrieverConfig,
    ResponseConfig
)
from rag.node_parser.text.sentence import SentenceSplitter
from rag.callbacks import CallbackManager
from rag.core.service_context import ServiceContext
from rag.engine.retriever_engine import RetrieverEngine
from rag.retrievers.sparse.bm25 import BM25Retriever


logger = logging.getLogger(__name__)

class BM25Pipeline:
    def __init__(
        self,
        splitter_config: NodeParserConfig,
        index_retriver_config: RetrieverConfig,
        response_config: ResponseConfig,
    ) -> None:
        self.splitter_config = splitter_config
        self.index_retriver_config = index_retriver_config
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

        self.service_context = ServiceContext(
            node_parser=node_parser,
            callback_manager=callback_manager,
        )
              

    def main(self, query, documents):

        """Wrap time"""
        start_build_collection_index = int(round(time.time() * 1000))

        # TODO: Get nodes from documents
        nodes = []
        for each_parser in self.service_context.transformations: 
            parsing_nodes = each_parser.get_nodes_from_documents(
                documents=documents,
                show_progress=True,
            )
            nodes.extend(parsing_nodes)

        end_build_collection_index = int(round(time.time() * 1000))
        print(f"Time for build collection and index: {end_build_collection_index - start_build_collection_index} ms")

        """Wrap for loading retriever and search"""
        start_build_retrieve_search = int(round(time.time() * 1000))

        #TODO: build retriever
        """
        tokenizer of bm25 retriever use remove_stopwords() function, 
        which different from the tokenizer of node_parser
        """
        retriever = BM25Retriever(
            nodes= nodes,
            similarity_top_k= self.index_retriver_config.similarity_top_k,
            callback_manager= self.service_context.callback_manager,
        )

        #TODO: assemble query engine
        query_engine = RetrieverEngine(
            retriever= retriever,
            callback_manager=self.service_context.callback_manager,
        )
        #TODO: query
        nodes = query_engine.retrieve(query)

        end_build_retrieve_search = int(round(time.time() * 1000))
        print(f"Time for build retriever and search: {end_build_retrieve_search - start_build_retrieve_search} ms")

        return nodes