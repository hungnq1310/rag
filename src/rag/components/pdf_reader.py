from typing import (
     Optional,
     List,
     Union,
     Dict,
)

from rag.entity.base_loader import PDFLoader_Base

class PyPDFLoader(PDFLoader_Base):
    """Load PDF using pypdf into list of documents.

    Loader chunks by page and stores page numbers in metadata.
    """

    def __init__(
        self,
        file_path: str,
        password: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict] = None,
        extract_images: bool = False,
    ) -> None:
        """Initialize with a file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )
        super().__init__(file_path, headers=headers)
        self.parser = PyPDFParser(password=password, extract_images=extract_images)


[docs]    def load(self) -> List[Document]:
        """Load given path as pages."""
        return list(self.lazy_load())


[docs]    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.web_path)
        else:
            blob = Blob.from_path(self.file_path)
        yield from self.parser.parse(blob)


class PDFMinerLoader(PDFLoader_Base):
    """Load `PDF` files using `PDFMiner`."""

    def __init__(
        self,
        file_path: str,
        *,
        headers: Optional[Dict] = None,
        extract_images: bool = False,
        concatenate_pages: bool = True,
    ) -> None:
        """Initialize with file path.

        Args:
            extract_images: Whether to extract images from PDF.
            concatenate_pages: If True, concatenate all PDF pages into one a single
                               document. Otherwise, return one document per page.
        """
        try:
            from pdfminer.high_level import extract_text  # noqa:F401
        except ImportError:
            raise ImportError(
                "`pdfminer` package not found, please install it with "
                "`pip install pdfminer.six`"
            )

        super().__init__(file_path, headers=headers)
        self.parser = PDFMinerParser(
            extract_images=extract_images, concatenate_pages=concatenate_pages
        )