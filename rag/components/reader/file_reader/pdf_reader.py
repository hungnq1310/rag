"""Simple reader that reads files of different formats from a directory."""
import logging
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional

from rag.entity.reader.base_reader import BaseReader
from rag.entity.node.base_node import Document

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
        # get all pdf files from directory
        pdf_files: List[Path] = glob.glob(os.path.join(file_path, "*.pdf"))

        docs = []
        for each_file in pdf_files:
            with open(each_file, "rb") as fp:
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
                metadata = {"num_pages": num_pages, "file_name": each_file.name}
                if extra_info is not None:
                    metadata.update(extra_info)
            # apply Document class
            docs.append(Document(text=text, metadata=metadata))
        return docs
        

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
        # get all pdf files from directory
        pdf_files: List[Path] = glob.glob(file_path + "/*.pdf")
        docs = []
        for each_file in pdf_files:
            text = extract_text(each_file)
            metadata = {"file_name": each_file.name}
            if extra_info is not None:
                metadata.update(extra_info)
            docs.append(Document(page_content=text, metadata=metadata))
        return docs