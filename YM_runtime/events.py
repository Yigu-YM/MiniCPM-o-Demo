"""Runtime event definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from YM_runtime.types import StageName


class EventType(str, Enum):
    """Events emitted by runtime services and backend stages."""

    SESSION_CREATED = "session_created"
    SESSION_PREPARED = "session_prepared"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    TEXT_DELTA = "text_delta"
    AUDIO_DELTA = "audio_delta"
    LISTEN = "listen"
    SPEAK = "speak"
    FINALIZED = "finalized"
    STOPPED = "stopped"
    CLEANED = "cleaned"
    ERROR = "error"


@dataclass(frozen=True)
class RuntimeEvent:
    """One event emitted by a runtime operation."""

    event_type: EventType
    session_id: Optional[str] = None
    stage_name: Optional[StageName] = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

