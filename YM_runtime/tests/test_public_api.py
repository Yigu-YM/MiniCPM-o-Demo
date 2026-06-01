import inspect


def test_public_api_exports_runtime_contracts():
    import YM_runtime as ym

    expected_exports = {
        "BackendCapabilities",
        "BackendInfo",
        "BackendKind",
        "EventType",
        "GenerateConfig",
        "InferenceMode",
        "MediaInput",
        "RuntimeBackend",
        "RuntimeEvent",
        "RuntimeService",
        "SessionSpec",
        "SessionState",
        "StageName",
        "StageRequest",
        "StageResult",
    }

    assert expected_exports.issubset(set(ym.__all__))
    for name in expected_exports:
        assert hasattr(ym, name)


def test_stage_contract_names_match_first_backend_alignment_scope():
    from YM_runtime import StageName

    assert [stage.value for stage in StageName] == [
        "prepare_session",
        "normalize_input",
        "vision_encode",
        "audio_encode",
        "llm_prefill",
        "llm_decode",
        "tts_decode",
        "duplex_decide",
        "finalize_unit",
        "cleanup_session",
    ]


def test_backend_protocol_exposes_fine_grained_stage_entrypoint():
    from YM_runtime import RuntimeBackend, StageResult

    run_stage = inspect.signature(RuntimeBackend.run_stage)

    assert list(run_stage.parameters) == ["self", "request"]
    assert run_stage.return_annotation is StageResult


def test_runtime_service_exposes_demo_adapter_entrypoints():
    from YM_runtime import RuntimeService

    expected_methods = {
        "load",
        "chat",
        "streaming_prefill",
        "streaming_generate",
        "duplex_prepare",
        "duplex_step",
        "duplex_stop",
        "cleanup_session",
    }

    assert expected_methods.issubset(set(dir(RuntimeService)))

