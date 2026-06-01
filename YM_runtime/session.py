"""Runtime session definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from YM_runtime.types import InferenceMode


class SessionState(str, Enum):
    """Lifecycle states controlled by the runtime."""

    CREATED = "created"
    PREPARED = "prepared"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    CLEANED = "cleaned"
    ERROR = "error"


@dataclass(frozen=True)
class SessionSpec:
    """Session creation and preparation parameters."""

    mode: InferenceMode
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    ref_audio_path: Optional[str] = None
    prompt_wav_path: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

