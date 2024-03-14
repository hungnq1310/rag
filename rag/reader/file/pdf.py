"""Simple reader that reads files of different formats from a directory."""
import logging
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional

from rag.reader.base_reader import BaseReader
from rag.node.base_node import Document

logger = logging.getLogger(__name__)

class PyPDFReader(BaseReader):
    """PDF parser."""

    def load_data(
        self, file_path: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required to read PDF files: `pip install pypdf`"
            )

        with open(file_path, "rb") as fp:
            # Create a PDF object
            pdf = pypdf.PdfReader(fp)

            # Get the number of pages in the PDF document
            num_pages = len(pdf.pages)

            # Iterate over every page
            pages_text = []
            for page in range(num_pages):
                # Extract the text from the page
                pages_text.append(pdf.pages[page].extract_text())
            # concatenate all pages to one long text.
            text = "".join(pages_text)
            # add description
            metadata = {"num_pages": num_pages, "file_name": file_path.name}
            if extra_info is not None:
                metadata.update(extra_info)

        return [Document(text=text, metadata=metadata)]
        

class PDFMinerLoader(BaseReader):
    """Load `PDF` files using `PDFMiner`."""

    def load_data(
        self, file_path: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        try:
            from pdfminer.high_level import extract_text  # noqa:F401
        except ImportError:
            raise ImportError(
                "`pdfminer` package not found, please install it with "
                "`pip install pdfminer.six`"
            )
        text = extract_text(file_path)
        metadata = {"file_name": file_path.name}
        if extra_info is not None:
            metadata.update(extra_info)
        return [Document(text=text, metadata=metadata)]