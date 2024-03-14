from typing import List
from pathlib import Path
from src.reader.base_reader import BaseReader
from src.node.base_node import Document


class TxtReader(BaseReader):
    """ Txt parser """

    def load_data(self, file: Path, extra_info=None) -> List[Document]:
        # load_data returns a list of Document objects
        with open(file, "r") as f:
            text = f.read()
        metadata = {"file_name": file.name}
        if extra_info is not None:
            metadata.update(extra_info)
        return [Document(text=text, extra_info=metadata)]