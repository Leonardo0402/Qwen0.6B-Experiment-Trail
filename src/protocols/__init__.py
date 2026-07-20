"""P4.1b Protocol abstraction layer."""
from src.protocols.base import ProtocolBase, ProtocolDiagnostics
from src.protocols.json_protocol import JsonProtocol
from src.protocols.tag_protocol import TagProtocol
from src.protocols.dsl_protocol import DslProtocol

__all__ = [
    "ProtocolBase", "ProtocolDiagnostics",
    "JsonProtocol", "TagProtocol", "DslProtocol",
]
