from typing import Set
from .base import BaseKeywordTableIndex
from .utils import simple_extract_keywords

class SimpleKeywordTableIndex(BaseKeywordTableIndex):
    """Simple Keyword Table Index.

    This index uses a simple regex extractor to extract keywords from the text.

    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        return simple_extract_keywords(text, self.max_keywords_per_chunk)


