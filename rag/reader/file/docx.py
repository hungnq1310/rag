from pathlib import Path
from typing import Optional, List, Dict

from src.node.base_node import Document
from src.reader.base_reader import BaseReader


class DocxReader(BaseReader):
    """Docx parser."""

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "docx2txt is required to read Microsoft Word files: "
                "`pip install docx2txt`"
            )

        text = docx2txt.process(file)
        metadata = {"file_name": file.name}
        if extra_info is not None:
            metadata.update(extra_info)

        return [Document(text=text, extra_info=metadata)]
