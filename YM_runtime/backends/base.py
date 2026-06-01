"""Backend contracts for PyTorch and remote inference implementations."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from YM_runtime.events import RuntimeEvent
from YM_runtime.session import SessionSpec
from YM_runtime.types import BackendKind, InferenceMode, StageName, StageRequest, StageResult


@dataclass(frozen=True)
class BackendCapabilities:
    """Backend capability declaration used for runtime routing."""

    supported_modes: tuple[InferenceMode, ...] = (
        InferenceMode.CHAT,
        InferenceMode.HALF_DUPLEX,
        InferenceMode.DUPLEX,
    )
    supported_stages: tuple[StageName, ...] = tuple(StageName)
    supports_stage_pipeline: bool = True
    supports_streaming: bool = True
    supports_duplex: bool = True
    supports_remote_session: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendInfo:
    """Backend identity and capability metadata."""

    name: str
    kind: BackendKind
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RuntimeBackend(Protocol):
    """Low-level backend API implemented by PyTorch and remote backends."""

    @property
    def info(self) -> BackendInfo:
        """Return backend identity and capabilities."""

    def load(self) -> None:
        """Load local resources or establish remote connectivity."""

    def close(self) -> None:
        """Close backend resources."""

    def create_session(self, spec: SessionSpec) -> RuntimeEvent:
        """Create or prepare a backend session."""

    def run_stage(self, request: StageRequest) -> StageResult:
        """Run one fine-grained inference stage."""

    def stream_stage(self, request: StageRequest) -> Iterable[RuntimeEvent]:
        """Run one streaming stage and yield runtime events."""

    def cleanup_session(self, session_id: str) -> RuntimeEvent:
        """Release resources for a backend session."""

