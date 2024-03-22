import logging 
import time

from transformers import AutoTokenizer

from rag.config.schema import (
    SpiltterConfig,
    IndexRetrieverConfig,
    ResponseConfig
)
from rag.node_parser import SentenceSplitter
from rag.callbacks import CallbackManager
from rag.core.service_context import ServiceContext
from rag.engine.retriever_engine import RetrieverQueryEngine
from rag.synthesizer.utils import get_response_synthesizer
from rag.indices.keyword_table.simple import SimpleKeywordTableIndex


logger = logging.getLogger(__name__)

class SimplKeywordeTablePipeline:
    def __init__(
        self,
        splitter_config: SpiltterConfig,
        index_retriver_config: IndexRetrieverConfig,
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

        # TODO: build index
        keyword_table_index = SimpleKeywordTableIndex(
            nodes=nodes,
            service_context= self.service_context,
            max_keywords_per_chunk=self.index_retriver_config.max_keywords_per_chunk,
            use_async=self.index_retriver_config.use_async,
            show_progress=self.index_retriver_config.show_progress,
        )

        end_build_collection_index = int(round(time.time() * 1000))
        print(f"Time for build collection and index: {end_build_collection_index - start_build_collection_index} ms")

        """Wrap for loading retriever and search"""
        start_build_retrieve_search = int(round(time.time() * 1000))

        #TODO: build retriever
        retriever = keyword_table_index.as_retriever(
            retriever_mode= self.index_retriver_config.keyword_table_mode,
            max_keywords_per_query= self.index_retriver_config.max_keywords_per_query,
            num_chunks_per_query= self.index_retriver_config.num_chunks_per_query,
            use_async= self.index_retriver_config.use_async,
            show_progress= self.index_retriver_config.show_progress,
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


        #TODO: assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever= retriever,
            response_synthesizer= response_synthesizer,
        )
        #TODO: query
        nodes = query_engine.retrieve(query)

        end_build_retrieve_search = int(round(time.time() * 1000))
        print(f"Time for build retriever and search: {end_build_retrieve_search - start_build_retrieve_search} ms")

        return nodes