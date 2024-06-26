"""Utilities for GPT indices."""
import logging
import re
from hashlib import sha256
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, TYPE_CHECKING

from rag.node.base_node import BaseNode, MetadataMode
from rag.rag_utils.utils import globals_helper, truncate_text
from rag.vector_stores.base_vector import VectorStoreQueryResult
from rag.schema.component import TransformComponent


if TYPE_CHECKING:
    from rag.embeddings.base_embeddings import BaseEmbedding

_logger = logging.getLogger(__name__)


def get_sorted_node_list(node_dict: Dict[int, BaseNode]) -> List[BaseNode]:
    """Get sorted node list. Used by tree-strutured indices."""
    sorted_indices = sorted(node_dict.keys())
    return [node_dict[index] for index in sorted_indices]


def extract_numbers_given_response(response: str, n: int = 1) -> Optional[List[int]]:
    """Extract number given the GPT-generated response.

    Used by tree-structured indices.

    """
    numbers = re.findall(r"\d+", response)
    if len(numbers) == 0:
        return None
    else:
        return numbers[:n]


def expand_tokens_with_subtokens(tokens: Set[str]) -> Set[str]:
    """Get subtokens from a list of tokens., filtering for stopwords."""
    results = set()
    for token in tokens:
        results.add(token)
        sub_tokens = re.findall(r"\w+", token)
        if len(sub_tokens) > 1:
            results.update({w for w in sub_tokens if w not in globals_helper.stopwords})

    return results


def log_vector_store_query_result(
    result: VectorStoreQueryResult, logger: Optional[logging.Logger] = None
) -> None:
    """Log vector store query result."""
    logger = logger or _logger

    assert result.ids is not None
    assert result.nodes is not None
    similarities = (
        result.similarities
        if result.similarities is not None and len(result.similarities) > 0
        else [1.0 for _ in result.ids]
    )

    fmt_txts = []
    for node_idx, node_similarity, node in zip(result.ids, similarities, result.nodes):
        fmt_txt = f"> [Node {node_idx}] [Similarity score: \
            {float(node_similarity):.6}] {truncate_text(node.get_content(), 100)}"
        fmt_txts.append(fmt_txt)
    top_k_node_text = "\n".join(fmt_txts)
    logger.debug(f"> Top {len(result.nodes)} nodes:\n{top_k_node_text}")


def default_format_node_batch_fn(
    summary_nodes: List[BaseNode],
) -> str:
    """Default format node batch function.

    Assign each summary node a number, and format the batch of nodes.

    """
    fmt_node_txts = []
    for idx in range(len(summary_nodes)):
        number = idx + 1
        fmt_node_txts.append(
            f"Document {number}:\n"
            f"{summary_nodes[idx].get_content(metadata_mode=MetadataMode.LLM)}"
        )
    return "\n\n".join(fmt_node_txts)


def default_parse_choice_select_answer_fn(
    answer: str, num_choices: int, raise_error: bool = False
) -> Tuple[List[int], List[float]]:
    """Default parse choice select answer function."""
    answer_lines = answer.split("\n")
    answer_nums = []
    answer_relevances = []
    for answer_line in answer_lines:
        line_tokens = answer_line.split(",")
        if len(line_tokens) != 2:
            if not raise_error:
                continue
            else:
                raise ValueError(
                    f"Invalid answer line: {answer_line}. "
                    "Answer line must be of the form: "
                    "answer_num: <int>, answer_relevance: <float>"
                )
        answer_num = int(line_tokens[0].split(":")[1].strip())
        if answer_num > num_choices:
            continue
        answer_nums.append(answer_num)
        answer_relevances.append(float(line_tokens[1].split(":")[1].strip()))
    return answer_nums, answer_relevances


def embed_nodes(
    nodes: Sequence[BaseNode], embed_model: "BaseEmbedding", show_progress: bool = False
) -> Dict[str, List[float]]:
    """Get embeddings of the given nodes, run embedding model if necessary.

    Args:
        nodes (Sequence[BaseNode]): The nodes to embed.
        embed_model (BaseEmbedding): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    Returns:
        Dict[str, List[float]]: A map from node id to embedding.
    """
    id_to_embed_map: Dict[str, List[float]] = {}

    texts_to_embed = []
    ids_to_embed = []
    for node in nodes:
        if node.embedding is None:
            ids_to_embed.append(node.node_id)
            texts_to_embed.append(node.get_content(metadata_mode=MetadataMode.EMBED))
        else:
            id_to_embed_map[node.node_id] = node.embedding

    new_embeddings = embed_model.get_text_embedding_batch(
        texts_to_embed, show_progress=show_progress
    )

    for new_id, text_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id] = text_embedding

    return id_to_embed_map



async def async_embed_nodes(
    nodes: Sequence[BaseNode], embed_model: "BaseEmbedding", show_progress: bool = False
) -> Dict[str, List[float]]:
    """Async get embeddings of the given nodes, run embedding model if necessary.

    Args:
        nodes (Sequence[BaseNode]): The nodes to embed.
        embed_model (BaseEmbedding): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    Returns:
        Dict[str, List[float]]: A map from node id to embedding.
    """
    id_to_embed_map: Dict[str, List[float]] = {}

    texts_to_embed = []
    ids_to_embed = []
    for node in nodes:
        if node.embedding is None:
            ids_to_embed.append(node.node_id)
            texts_to_embed.append(node.get_content(metadata_mode=MetadataMode.EMBED))
        else:
            id_to_embed_map[node.node_id] = node.embedding

    new_embeddings = await embed_model.aget_text_embedding_batch(
        texts_to_embed, show_progress=show_progress
    )

    for new_id, text_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id] = text_embedding

    return id_to_embed_map

def remove_unstable_values(s: str) -> str:
    """Remove unstable key/value pairs.

    Examples include:
    - <__main__.Test object at 0x7fb9f3793f50>
    - <function test_fn at 0x7fb9f37a8900>
    """
    pattern = r"<[\w\s_\. ]+ at 0x[a-z0-9]+>"
    return re.sub(pattern, "", s)


def get_transformation_hash(
    nodes: List[BaseNode], transformation: TransformComponent
) -> str:
    """Get the hash of a transformation."""
    nodes_str = "".join(
        [str(node.get_content(metadata_mode=MetadataMode.ALL)) for node in nodes]
    )

    transformation_dict = transformation.to_dict()
    transform_string = remove_unstable_values(str(transformation_dict))

    return sha256((nodes_str + transform_string).encode("utf-8")).hexdigest()


def run_transformations(
    nodes: List[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    cache_collection: Optional[str] = None,
    **kwargs: Any,
) -> List[BaseNode]:
    """Run a series of transformations on a set of nodes.

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.
    """
    if not in_place:
        nodes = list(nodes)

    for transform in transformations:
        nodes = transform(nodes, **kwargs)

    return nodes