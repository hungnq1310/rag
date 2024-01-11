from typing import Dict, Type

from rag.entity.node_parser import NodeParser
from rag.components.node_parser.relationship.hierarchical import HierarchicalNodeParser
from rag.components.node_parser.text.sentence import SentenceSplitter
from rag.components.node_parser.text.sentence_window import SentenceWindowNodeParser
from rag.components.node_parser.text.token import TokenTextSplitter

all_node_parsers: Dict[str, Type[NodeParser]] = {
    HierarchicalNodeParser.class_name(): HierarchicalNodeParser,
    SentenceSplitter.class_name(): SentenceSplitter,
    TokenTextSplitter.class_name(): TokenTextSplitter,
    SentenceWindowNodeParser.class_name(): SentenceWindowNodeParser,
}


def load_parser(
    data: dict,
) -> NodeParser:
    if isinstance(data, NodeParser):
        return data
    parser_name = data.get("class_name", None)
    if parser_name is None:
        raise ValueError("Parser loading requires a class_name")

    if parser_name not in all_node_parsers:
        raise ValueError(f"Invalid parser name: {parser_name}")
    else:
        return all_node_parsers[parser_name].from_dict(data)