"""Neutral runtime types shared by services and backends."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


class BackendKind(str, Enum):
    """Backend implementation category."""

    PYTORCH = "pytorch"
    REMOTE = "remote"
    MOCK = "mock"


class InferenceMode(str, Enum):
    """Runtime mode compatible with the current MiniCPM-o demo surfaces."""

    CHAT = "chat"
    HALF_DUPLEX = "half_duplex"
    DUPLEX = "duplex"


class StageName(str, Enum):
    """Fine-grained backend stage names for remote backend alignment."""

    PREPARE_SESSION = "prepare_session"
    NORMALIZE_INPUT = "normalize_input"
    VISION_ENCODE = "vision_encode"
    AUDIO_ENCODE = "audio_encode"
    LLM_PREFILL = "llm_prefill"
    LLM_DECODE = "llm_decode"
    TTS_DECODE = "tts_decode"
    DUPLEX_DECIDE = "duplex_decide"
    FINALIZE_UNIT = "finalize_unit"
    CLEANUP_SESSION = "cleanup_session"


@dataclass(frozen=True)
class MediaInput:
    """One normalized media input item flowing through the runtime."""

    modality: str
    data: Any
    mime_type: Optional[str] = None
    sample_rate: Optional[int] = None
    timestamp_ms: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerateConfig:
    """Generation controls shared by streaming and duplex paths."""

    max_new_tokens: int = 256
    do_sample: bool = True
    generate_audio: bool = True
    length_penalty: float = 1.1
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    force_listen: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageRequest:
    """Request to run one fine-grained backend stage."""

    stage_name: StageName
    session_id: Optional[str] = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    """Result returned by a backend stage."""

    stage_name: StageName
    success: bool = True
    payload: Mapping[str, Any] = field(default_factory=dict)
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    error: Optional[str] = None

