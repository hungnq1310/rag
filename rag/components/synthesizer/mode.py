from enum import Enum


class ResponseMode(str, Enum):
    """Response modes of the response builder (and synthesizer)."""

    SIMPLE_SUMMARIZE = "simple_summarize"
    """
    Merge all text chunks into one, and make a LLM call.
    This will fail if the merged text chunk exceeds the context window size.
    """

    TREE_SUMMARIZE = "tree_summarize"
    """
    Build a tree index over the set of candidate nodes, with a summary prompt seeded \
    with the query.
    The tree is built in a bottoms-up fashion, and in the end the root node is \
    returned as the response
    """

    GENERATION = "generation"
    """Ignore context, just use LLM to generate a response."""

    NO_TEXT = "no_text"
    """Return the retrieved context nodes, without synthesizing a final response."""
