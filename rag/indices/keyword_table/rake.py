from typing import Set

from .base import BaseKeywordTableIndex
from .utils import rake_extract_keywords

class RAKEKeywordTableIndex(BaseKeywordTableIndex):
    """RAKE Keyword Table Index.

    This index uses a RAKE keyword extractor to extract keywords from the text.

    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        return rake_extract_keywords(text, max_keywords=self.max_keywords_per_chunk)
