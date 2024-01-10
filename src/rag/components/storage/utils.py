from rag.entity.storage.chat_store.base import BaseChatStore
from rag.entity.storage.chat_store.simple_chat_store import SimpleChatStore
from llama_index.constants import DATA_KEY, TYPE_KEY
from llama_index.data_structs.data_structs import IndexStruct
from llama_index.data_structs.registry import INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS

RECOGNIZED_CHAT_STORES = {
    SimpleChatStore.class_name(): SimpleChatStore,
}


def load_chat_store(data: dict) -> BaseChatStore:
    """Load a chat store from a dict."""
    chat_store_name = data.get("class_name", None)
    if chat_store_name is None:
        raise ValueError("ChatStore loading requires a class_name")

    if chat_store_name not in RECOGNIZED_CHAT_STORES:
        raise ValueError(f"Invalid ChatStore name: {chat_store_name}")

    return RECOGNIZED_CHAT_STORES[chat_store_name].from_dict(data)

def index_struct_to_json(index_struct: IndexStruct) -> dict:
    return {
        TYPE_KEY: index_struct.get_type(),
        DATA_KEY: index_struct.to_json(),
    }


def json_to_index_struct(struct_dict: dict) -> IndexStruct:
    type = struct_dict[TYPE_KEY]
    data_dict = struct_dict[DATA_KEY]
    cls = INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS[type]
    try:
        return cls.from_json(data_dict)
    except TypeError:
        return cls.from_dict(data_dict)
    
from llama_index.constants import DATA_KEY, TYPE_KEY
from llama_index.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    IndexNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)


def doc_to_json(doc: BaseNode) -> dict:
    return {
        DATA_KEY: doc.dict(),
        TYPE_KEY: doc.get_type(),
    }


def json_to_doc(doc_dict: dict) -> BaseNode:
    doc_type = doc_dict[TYPE_KEY]
    data_dict = doc_dict[DATA_KEY]
    doc: BaseNode

    if "extra_info" in data_dict:
        return legacy_json_to_doc(doc_dict)
    else:
        if doc_type == Document.get_type():
            doc = Document.parse_obj(data_dict)
        elif doc_type == ImageDocument.get_type():
            doc = ImageDocument.parse_obj(data_dict)
        elif doc_type == TextNode.get_type():
            doc = TextNode.parse_obj(data_dict)
        elif doc_type == ImageNode.get_type():
            doc = ImageNode.parse_obj(data_dict)
        elif doc_type == IndexNode.get_type():
            doc = IndexNode.parse_obj(data_dict)
        else:
            raise ValueError(f"Unknown doc type: {doc_type}")

        return doc


def legacy_json_to_doc(doc_dict: dict) -> BaseNode:
    """Todo: Deprecated legacy support for old node versions."""
    doc_type = doc_dict[TYPE_KEY]
    data_dict = doc_dict[DATA_KEY]
    doc: BaseNode

    text = data_dict.get("text", "")
    metadata = data_dict.get("extra_info", {}) or {}
    id_ = data_dict.get("doc_id", None)

    relationships = data_dict.get("relationships", {})
    relationships = {
        NodeRelationship(k): RelatedNodeInfo(node_id=v)
        for k, v in relationships.items()
    }

    if doc_type == Document.get_type():
        doc = Document(
            text=text, metadata=metadata, id=id_, relationships=relationships
        )
    elif doc_type == TextNode.get_type():
        doc = TextNode(
            text=text, metadata=metadata, id=id_, relationships=relationships
        )
    elif doc_type == ImageNode.get_type():
        image = data_dict.get("image", None)
        doc = ImageNode(
            text=text,
            metadata=metadata,
            id=id_,
            relationships=relationships,
            image=image,
        )
    elif doc_type == IndexNode.get_type():
        index_id = data_dict.get("index_id", None)
        doc = IndexNode(
            text=text,
            metadata=metadata,
            id=id_,
            relationships=relationships,
            index_id=index_id,
        )
    else:
        raise ValueError(f"Unknown doc type: {doc_type}")

    return doc