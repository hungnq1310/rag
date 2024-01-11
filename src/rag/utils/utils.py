from typing import List, Dict, Optional, Generator, cast
from itertools import islice
import sys
from pathlib import Path

# Sample text from llama_index's readme
SAMPLE_TEXT = """
Context
LLMs are a phenomenal piece of technology for knowledge generation and reasoning.
They are pre-trained on large amounts of publicly available data.
How do we best augment LLMs with our own private data?
We need a comprehensive toolkit to help perform this data augmentation for LLMs.

Proposed Solution
That's where LlamaIndex comes in. LlamaIndex is a "data framework" to help
you build LLM  apps. It provides the following tools:

Offers data connectors to ingest your existing data sources and data formats
(APIs, PDFs, docs, SQL, etc.)
Provides ways to structure your data (indices, graphs) so that this data can be
easily used with LLMs.
Provides an advanced retrieval/query interface over your data:
Feed in any LLM input prompt, get back retrieved context and knowledge-augmented output.
Allows easy integrations with your outer application framework
(e.g. with LangChain, Flask, Docker, ChatGPT, anything else).
LlamaIndex provides tools for both beginner users and advanced users.
Our high-level API allows beginner users to use LlamaIndex to ingest and
query their data in 5 lines of code. Our lower-level APIs allow advanced users to
customize and extend any module (data connectors, indices, retrievers, query engines,
reranking modules), to fit their needs.
"""

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."

_LLAMA_INDEX_COLORS = {
    "llama_pink": "38;2;237;90;200",
    "llama_blue": "38;2;90;149;237",
    "llama_turquoise": "38;2;11;159;203",
    "llama_lavender": "38;2;155;135;227",
}

_ANSI_COLORS = {
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "pink": "38;5;200",
}

def get_color_mapping(
    items: List[str], use_llama_index_colors: bool = True
) -> Dict[str, str]:
    """
    Get a mapping of items to colors.

    Args:
        items (List[str]): List of items to be mapped to colors.
        use_llama_index_colors (bool, optional): Flag to indicate
        whether to use LlamaIndex colors or ANSI colors.
            Defaults to True.

    Returns:
        Dict[str, str]: Mapping of items to colors.
    """
    if use_llama_index_colors:
        color_palette = _LLAMA_INDEX_COLORS
    else:
        color_palette = _ANSI_COLORS

    colors = list(color_palette.keys())
    return {item: colors[i % len(colors)] for i, item in enumerate(items)}

def print_text(text: str, color: Optional[str] = None, end: str = "") -> None:
    """
    Print the text with the specified color.

    Args:
        text (str): Text to be printed.
        color (str, optional): Color to be applied to the text. Supported colors are:
            llama_pink, llama_blue, llama_turquoise, llama_lavender,
            red, green, yellow, blue, magenta, cyan, pink.
        end (str, optional): String appended after the last character of the text.

    Returns:
        None
    """
    text_to_print = _get_colored_text(text, color) if color is not None else text
    print(text_to_print, end=end)

def _get_colored_text(text: str, color: str) -> str:
    """
    Get the colored version of the input text.

    Args:
        text (str): Input text.
        color (str): Color to be applied to the text.

    Returns:
        str: Colored version of the input text.
    """
    all_colors = {**_LLAMA_INDEX_COLORS, **_ANSI_COLORS}

    if color not in all_colors:
        return f"\033[1;3m{text}\033[0m"  # just bolded and italicized

    color = all_colors[color]

    return f"\033[1;3;{color}m{text}\033[0m"

"""Support for synthesizer"""

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."

"""support for service context"""
from typing import (
    Callable,
    Union,
    Any,
    Protocol,
    runtime_checkable,
    Iterable,
)
from functools import partial
import os

# global tokenizer
global_tokenizer: Optional[Callable[[str], list]] = None

# Global Tokenizer
@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str, *args: Any, **kwargs: Any) -> List[Any]:
        ...


def set_global_tokenizer(tokenizer: Union[Tokenizer, Callable[[str], list]]) -> None:

    if isinstance(tokenizer, Tokenizer):
        global_tokenizer = tokenizer.encode
    else:
        global_tokenizer = tokenizer


def get_tokenizer() -> Callable[[str], List]:

    if global_tokenizer is None:
        tiktoken_import_err = (
            "`tiktoken` package not found, please run `pip install tiktoken`"
        )
        try:
            import tiktoken
        except ImportError:
            raise ImportError(tiktoken_import_err)

        # set tokenizer cache temporarily
        should_revert = False
        if "TIKTOKEN_CACHE_DIR" not in os.environ:
            should_revert = True
            os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "_static/tiktoken_cache",
            )

        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokenizer = partial(enc.encode, allowed_special="all")
        set_global_tokenizer(tokenizer)

        if should_revert:
            del os.environ["TIKTOKEN_CACHE_DIR"]

    assert global_tokenizer is not None
    return global_tokenizer


def get_tqdm_iterable(items: Iterable, show_progress: bool, desc: str) -> Iterable:
    """
    Optionally get a tqdm iterable. Ensures tqdm.auto is used.
    """
    _iterator = items
    if show_progress:
        try:
            from tqdm.auto import tqdm

            return tqdm(items, desc=desc)
        except ImportError:
            pass
    return _iterator


def count_tokens(text: str) -> int:
    tokenizer = get_tokenizer()
    tokens = tokenizer(text)
    return len(tokens)


def get_cache_dir() -> str:
    """Locate a platform-appropriate cache directory for llama_index,
    and create it if it doesn't yet exist.
    """
    # User override
    if "LLAMA_INDEX_CACHE_DIR" in os.environ:
        path = Path(os.environ["LLAMA_INDEX_CACHE_DIR"])

    # Linux, Unix, AIX, etc.
    elif os.name == "posix" and sys.platform != "darwin":
        path = Path("/tmp/llama_index")

    # Mac OS
    elif sys.platform == "darwin":
        path = Path(os.path.expanduser("~"), "Library/Caches/llama_index")

    # Windows (hopefully)
    else:
        local = os.environ.get("LOCALAPPDATA", None) or os.path.expanduser(
            "~\\AppData\\Local"
        )
        path = Path(local, "llama_index")

    if not os.path.exists(path):
        os.makedirs(
            path, exist_ok=True
        )  # prevents https://github.com/jerryjliu/llama_index/issues/7362
    return str(path)

def concat_dirs(dirname: str, basename: str) -> str:
    """
    Append basename to dirname, avoiding backslashes when running on windows.

    os.path.join(dirname, basename) will add a backslash before dirname if
    basename does not end with a slash, so we make sure it does.
    """
    dirname += "/" if dirname[-1] != "/" else ""
    return os.path.join(dirname, basename)


def iter_batch(iterable: Union[Iterable, Generator], size: int) -> Iterable:
    """Iterate over an iterable in batches.

    >>> list(iter_batch([1,2,3,4,5], 3))
    [[1, 2, 3], [4, 5]]
    """
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, size))
        if len(b) == 0:
            break
        yield b

class GlobalsHelper:
    """Helper to retrieve globals.

    Helpful for global caching of certain variables that can be expensive to load.
    (e.g. tokenization)

    """

    _tokenizer: Optional[Callable[[str], List]] = None
    _stopwords: Optional[List[str]] = None

    @property
    def tokenizer(self) -> Callable[[str], List]:
        """Get tokenizer. TODO: Deprecated."""
        if self._tokenizer is None:
            tiktoken_import_err = (
                "`tiktoken` package not found, please run `pip install tiktoken`"
            )
            try:
                import tiktoken
            except ImportError:
                raise ImportError(tiktoken_import_err)
            enc = tiktoken.get_encoding("gpt2")
            self._tokenizer = cast(Callable[[str], List], enc.encode)
            self._tokenizer = partial(self._tokenizer, allowed_special="all")
        return self._tokenizer  # type: ignore

    @property
    def stopwords(self) -> List[str]:
        """Get stopwords."""
        if self._stopwords is None:
            try:
                import nltk
                from nltk.corpus import stopwords
            except ImportError:
                raise ImportError(
                    "`nltk` package not found, please run `pip install nltk`"
                )

            from llama_index.utils import get_cache_dir

            cache_dir = get_cache_dir()
            nltk_data_dir = os.environ.get("NLTK_DATA", cache_dir)

            # update nltk path for nltk so that it finds the data
            if nltk_data_dir not in nltk.data.path:
                nltk.data.path.append(nltk_data_dir)

            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", download_dir=nltk_data_dir)
            self._stopwords = stopwords.words("english")
        return self._stopwords


globals_helper = GlobalsHelper()