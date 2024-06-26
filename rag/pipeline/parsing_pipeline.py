"""Simple node parser."""
from logging import getLogger
from typing import Any, List, Union, Sequence
from dataclasses import dataclass
from transformers import AutoTokenizer

from rag.config.schema import (
    SpiltterConfig
)                               
from rag.node_parser.text import TokenTextSplitter
from rag.node_parser.text import SentenceSplitter
from rag.node.base_node import (
    BaseNode,
    Document,
)

logger = getLogger(__name__)

@dataclass
class SplitterMode:
    SENTENCE = "SentenceSplitter"
    TOKEN = "TokenTextSplitter"

class ParsingPipeline:
    """Define node parser.

    Splits a document into Nodes using a TextSplitter.

    Args:3
        text_splitter (Optional[TextSplitter]): text splitter
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """

    def __init__(self,
                 config: SpiltterConfig,
                 ) -> None:
        self.config = config
        self.text_splitter = self.load_text_splitter()
        self.parser = self.load_node_parser()

    def load_text_splitter(
        self,
    ) -> Union[SentenceSplitter, TokenTextSplitter, None]:
        text_splitter = None
        if self.config.splitter_mode == SplitterMode.SENTENCE:
            text_splitter = SentenceSplitter(
                # separator=" ", 
                separator=self.config.separator, 
                chunk_size=self.config.chunk_size, 
                chunk_overlap=self.config.chunk_overlap,
                # paragraph_separator="\n\n\n", secondary_chunking_regex="[^,.;。]+[,.;。]?",
                paragraph_separator=self.config.paragraph_separator, 
                secondary_chunking_regex=self.config.secondary_chunking_regex,
                # tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
                tokenizer=AutoTokenizer.from_pretrained(self.config.model_name_tokenizer).encode
            )
            logger.info("Using SentenceSplitter")
        elif self.config.splitter_mode == SplitterMode.TOKEN:
            text_splitter = TokenTextSplitter(
                separator=self.config.separator, 
                chunk_size=self.config.chunk_size, 
                chunk_overlap=self.config.chunk_overlap,
                # backup_separators=["\n"],
                backup_separators=self.config.backup_separators,
                tokenizer=AutoTokenizer.from_pretrained(self.config.model_name_tokenizer).encode
            )
            logger.info("Using TokenTextSplitter")
        return text_splitter

    def load_node_parser(self):
        if not self.text_splitter:
            return SentenceSplitter.from_defaults(
                chunk_size=self.config.chunk_size, 
                chunk_overlap=self.config.chunk_overlap
            )
        return SentenceSplitter.from_defaults(
            text_splitter=self.text_splitter
        )

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        return self.parser.get_nodes_from_documents(
            documents=documents,
            show_progress=show_progress
        )