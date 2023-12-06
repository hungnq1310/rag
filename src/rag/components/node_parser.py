"""Simple node parser."""
from logging import getLogger
from typing import Any, List, Union, Sequence
import tiktoken

from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter
from llama_index.text_splitter import SentenceSplitter
from llama_index.text_splitter import CodeSplitter
from llama_index.schema import (
    BaseNode,
    Document,
)

logger = getLogger(__name__)

class TextNodesParser:
    """Define node parser.

    Splits a document into Nodes using a TextSplitter.

    Args:3
        text_splitter (Optional[TextSplitter]): text splitter
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    def __init__(self,
                 config: NodeParserConfig,
                 params: NodeParserParams, 
                 ) -> None:
        self.config = config
        self.params = params
        self.parser = self.load_node_parser()
        self.text_splitter = self.load_text_splitter()

    def load_text_splitter(
        self,
    ) -> Union[SentenceSplitter, TokenTextSplitter, CodeSplitter, None]:
        text_splitter = None
        if self.text_splitter == "SentenceSplitter":
            text_splitter = SentenceSplitter(
                # separator=" ", 
                separator=self.params.separator, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap,
                # paragraph_separator="\n\n\n", secondary_chunking_regex="[^,.;。]+[,.;。]?",
                paragraph_separator=self.params.paragraph_separator, 
                secondary_chunking_regex=self.params.secondary_chunking_regex,
                # tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
                tokenizer=tiktoken.encoding_for_model(self.params.model_name).encode
            )
            logger.info("Using SentenceSplitter")
        elif self.text_splitter == "TokenTextSplitter":
            text_splitter = TokenTextSplitter(
                separator=self.params.separator, 
                chunk_size=self.params.chunk_size, 
                chunk_overlap=self.params.chunk_overlap,
                # backup_separators=["\n"],
                backup_separators=self.params.backup_separators,
                tokenizer=tiktoken.encoding_for_model(self.params.model_name).encode
            )
            logger.info("Using TokenTextSplitter")
        else:
            text_splitter = CodeSplitter(
                # language="python", chunk_lines=40, chunk_lines_overlap=15, max_chars=1500,
                language=self.params.language, 
                chunk_lines=self.params.chunk_lines, 
                chunk_lines_overlap=self.params.chunk_lines_overlap, 
                max_chars=self.params.max_chars,
            )
            logger.info("Using CodeSplitter")
        return text_splitter

    def load_node_parser(self):
        if self.text_splitter:
            return SimpleNodeParser.from_defaults(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return SimpleNodeParser.from_defaults(text_splitter=self.text_splitter)

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        return self.parser.get_nodes_from_documents(documents=documents,show_progress=show_progress)