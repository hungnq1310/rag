"""Utils for keyword table."""

import re
from typing import Optional, Set

import pandas as pd

from rag.entity.indices.utils import expand_tokens_with_subtokens
from rag.utils.utils import globals_helper


def simple_extract_keywords(
    text_chunk: str, max_keywords: Optional[int] = None, filter_stopwords: bool = True
) -> Set[str]:
    """Extract keywords with simple algorithm."""
    tokens = [t.strip().lower() for t in re.findall(r"\w+", text_chunk)]
    if filter_stopwords:
        tokens = [t for t in tokens if t not in globals_helper.stopwords]
    value_counts = pd.Series(tokens).value_counts()
    keywords = value_counts.index.tolist()[:max_keywords]
    return set(keywords)


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
