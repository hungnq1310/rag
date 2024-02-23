from rag.callbacks.callback_manager import CallbackManager
from rag.callbacks.types import EventPayload, CBEventType

__all__ = [
    "CallbackManager",
    "EventPayload",
    "CBEventType"
]


from  typing import Optional
from rag.callbacks.base_handler import BaseCallbackHandler
global_handler: Optional[BaseCallbackHandler] = None