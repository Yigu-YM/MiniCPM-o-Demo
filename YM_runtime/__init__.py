"""Public API for the YM runtime package.

The runtime is intentionally defined in model-neutral terms. Demo workers
adapt MiniCPM-o request/response schemas into these contracts, while backends
implement either coarse PyTorch calls or fine-grained stage execution.
"""

from YM_runtime.api import RuntimeService
from YM_runtime.backends.base import BackendCapabilities, BackendInfo, RuntimeBackend
from YM_runtime.backends.mock import MockBackend
from YM_runtime.events import EventType, RuntimeEvent
from YM_runtime.runtime import RuntimeServiceImpl
from YM_runtime.session import SessionSpec, SessionState
from YM_runtime.types import (
    BackendKind,
    GenerateConfig,
    InferenceMode,
    MediaInput,
    StageName,
    StageRequest,
    StageResult,
)

__all__ = [
    "BackendCapabilities",
    "BackendInfo",
    "BackendKind",
    "EventType",
    "GenerateConfig",
    "InferenceMode",
    "MediaInput",
    "MockBackend",
    "RuntimeBackend",
    "RuntimeEvent",
    "RuntimeService",
    "RuntimeServiceImpl",
    "SessionSpec",
    "SessionState",
    "StageName",
    "StageRequest",
    "StageResult",
]

