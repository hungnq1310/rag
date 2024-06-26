"""Keyword-table based index.

Similar to a "hash table" in concept. LlamaIndex first tries
to extract keywords from the source text, and stores the
keywords as keys per item. It similarly extracts keywords
from the query text. Then, it tries to match those keywords to
existing keywords in the table.

"""

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Set, Union, TYPE_CHECKING

from rag.rag_utils.async_utils import run_async_tasks
from rag.node.base_node import BaseNode, MetadataMode
from rag.rag_utils.utils import get_tqdm_iterable
from rag.indices.data_struct import KeywordTable
from rag.indices.base_index import BaseIndex

from .utils import extract_keywords_given_response

from rag.constants.default_prompt import (
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)

if TYPE_CHECKING:
    from rag.retrievers.base import BaseRetriever
    from rag.prompt.base_prompt import BasePromptTemplate
    from rag.core.service_context import ServiceContext
    from rag.core.storage_context import StorageContext
    from rag.storage.docstore.base import RefDocInfo  


DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE


class KeywordTableRetrieverMode(str, Enum):
    DEFAULT = "default"
    SIMPLE = "simple"
    RAKE = "rake"


class BaseKeywordTableIndex(BaseIndex):
    """Base Keyword Table Index.

    This index extracts keywords from the text, and maps each
    keyword to the node(s) that it corresponds to. In this sense it mimics a
    "hash table". During index construction, the keyword table is constructed
    by extracting keywords from each node and creating an internal mapping.

    During query time, the keywords are extracted from the query text, and these
    keywords are used to index into the keyword table. The retrieved nodes
    are then used to answer the query.

    Args:
        keyword_extract_template (Optional[BasePromptTemplate]): A Keyword
            Extraction Prompt
            (see :ref:`Prompt-Templates`).
        use_async (bool): Whether to use asynchronous calls. Defaults to False.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.

    """

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[KeywordTable] = None,
        service_context: Optional["ServiceContext"] = None,
        storage_context: Optional["StorageContext"] = None,
        keyword_extract_template: Optional["BasePromptTemplate"] = None,
        max_keywords_per_chunk: int = 10,
        use_async: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.max_keywords_per_chunk = max_keywords_per_chunk
        self.keyword_extract_template = (
            keyword_extract_template or DEFAULT_KEYWORD_EXTRACT_TEMPLATE
        )
        # NOTE: Partially format keyword extract template here.
        self.keyword_extract_template = self.keyword_extract_template.partial_format(
            max_keywords=self.max_keywords_per_chunk
        )
        self._use_async = use_async
        super().__init__(
            nodes=nodes,
            index_struct=index_struct or KeywordTable(),
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs,
        )

    def as_retriever(
        self,
        retriever_mode: Union[
            str, KeywordTableRetrieverMode
        ] = KeywordTableRetrieverMode.DEFAULT,
        **kwargs: Any,
    ) -> "BaseRetriever":
        # NOTE: lazy import
        from rag.retrievers.sparse.keyword_retriever import (
            KeywordTableRetriever,
            KeywordTableRAKERetriever,
            KeywordTableSimpleRetriever,
        )

        if retriever_mode == KeywordTableRetrieverMode.DEFAULT:
            return KeywordTableRetriever(self, **kwargs)
        elif retriever_mode == KeywordTableRetrieverMode.SIMPLE:
            return KeywordTableSimpleRetriever(self, **kwargs)
        elif retriever_mode == KeywordTableRetrieverMode.RAKE:
            return KeywordTableRAKERetriever(self, **kwargs)
        else:
            raise ValueError(f"Unknown retriever mode: {retriever_mode}")

    @abstractmethod
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""

    async def _async_extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # by default just call sync version
        return self._extract_keywords(text)

    def _add_nodes_to_index(
        self,
        index_struct: KeywordTable,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        """Add document to index."""
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Extracting keywords from nodes"
        )
        for n in nodes_with_progress:
            keywords = self._extract_keywords(
                n.get_content(metadata_mode=MetadataMode.LLM)
            )
            index_struct.add_node(list(keywords), n)

    async def _async_add_nodes_to_index(
        self,
        index_struct: KeywordTable,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        """Add document to index."""
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Extracting keywords from nodes"
        )
        for n in nodes_with_progress:
            keywords = await self._async_extract_keywords(
                n.get_content(metadata_mode=MetadataMode.LLM)
            )
            index_struct.add_node(list(keywords), n)

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> KeywordTable:
        """Build the index from nodes."""
        # do simple concatenation
        index_struct: KeywordTable = self.index_struct # type: ignore
        if self._use_async:
            tasks = [
                self._async_add_nodes_to_index(index_struct, nodes, self._show_progress)
            ]
            run_async_tasks(tasks)
        else:
            self._add_nodes_to_index(index_struct, nodes, self._show_progress)

        return index_struct

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert nodes."""
        for n in nodes:
            keywords = self._extract_keywords(
                n.get_content(metadata_mode=MetadataMode.LLM)
            )
            self.index_struct.add_node(list(keywords), n) # type: ignore

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        # delete node from the keyword table
        keywords_to_delete = set()
        for keyword, existing_node_ids in self.index_struct.table.items(): # type: ignore
            if node_id in existing_node_ids:
                existing_node_ids.remove(node_id)
                if len(existing_node_ids) == 0:
                    keywords_to_delete.add(keyword)

        # delete keywords that have zero nodes
        for keyword in keywords_to_delete:
            del self.index_struct.table[keyword] # type: ignore

    @property
    def ref_doc_info(self) -> Dict[str, "RefDocInfo"]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        node_doc_ids_sets = list(self.index_struct.table.values()) # type: ignore
        node_doc_ids = list(set().union(*node_doc_ids_sets))
        nodes = self.docstore.get_nodes(node_doc_ids) # type: ignore

        all_ref_doc_info = {}
        for node in nodes:
            ref_node = node.source_node
            if not ref_node:
                continue

            ref_doc_info = self.docstore.get_ref_doc_info(ref_node.node_id)
            if not ref_doc_info:
                continue

            all_ref_doc_info[ref_node.node_id] = ref_doc_info
        return all_ref_doc_info


class KeywordTableIndex(BaseKeywordTableIndex):
    """Keyword Table Index.

    This index uses a GPT model to extract keywords from the text.

    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        response = self.service_context.llm.predict(
            self.keyword_extract_template,
            text=text,
        )
        return extract_keywords_given_response(response, start_token="KEYWORDS:")

    async def _async_extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        response = await self.service_context.llm.apredict(
            self.keyword_extract_template,
            text=text,
        )
        return extract_keywords_given_response(response, start_token="KEYWORDS:")
