"""Utils for keyword table."""
from typing import Set

from rag.entity.indices.utils import expand_tokens_with_subtokens


def extract_keywords_given_response(
    response: str, lowercase: bool = True, start_token: str = ""
) -> Set[str]:
    """Extract keywords given the GPT-generated response.

    Used by keyword table indices.
    Parses <start_token>: <word1>, <word2>, ... into [word1, word2, ...]
    Raises exception if response doesn't start with <start_token>
    """
    results = []
    response = response.strip()  # Strip newlines from responses.

    if response.startswith(start_token):
        response = response[len(start_token) :]

    keywords = response.split(",")
    for k in keywords:
        rk = k
        if lowercase:
            rk = rk.lower()
        results.append(rk.strip())

    # if keyword consists of multiple words, split into subwords
    # (removing stopwords)
    return expand_tokens_with_subtokens(set(results))
