"""Runtime-facing API used by demo adapters."""

from typing import Any, Iterable, Protocol

from YM_runtime.events import RuntimeEvent
from YM_runtime.session import SessionSpec
from YM_runtime.types import GenerateConfig, MediaInput


class RuntimeService(Protocol):
    """High-level runtime API exposed to the MiniCPM-o demo adapter."""

    def load(self) -> None:
        """Load the configured backend and prepare runtime resources."""

    def chat(self, request: Any) -> Any:
        """Run a turn-based chat request.

        The demo adapter may pass the existing MiniCPM-o ChatRequest here during
        the migration phase. A later standalone runtime can replace this with a
        neutral chat schema without changing backend contracts.
        """

    def streaming_prefill(self, session_id: str, request: Any) -> RuntimeEvent:
        """Prefill a half-duplex or streaming chat session."""

    def streaming_generate(
        self,
        session_id: str,
        config: GenerateConfig,
    ) -> Iterable[RuntimeEvent]:
        """Generate streaming text/audio events for an existing session."""

    def duplex_prepare(self, spec: SessionSpec) -> RuntimeEvent:
        """Prepare a full-duplex session."""

    def duplex_step(
        self,
        session_id: str,
        inputs: list[MediaInput],
        config: GenerateConfig,
    ) -> Iterable[RuntimeEvent]:
        """Run one duplex input unit through prefill, generate, and finalize."""

    def duplex_stop(self, session_id: str) -> RuntimeEvent:
        """Stop a duplex session."""

    def cleanup_session(self, session_id: str) -> RuntimeEvent:
        """Release backend resources associated with a session."""

