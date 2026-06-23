"""Tests for headroom.integrations.litellm_callback."""

from __future__ import annotations

import importlib
import inspect

import pytest


def _import_callback() -> type:
    # Import the module directly to avoid triggering headroom/integrations/__init__.py
    # which pulls in langchain and the native .so extension.
    spec = importlib.util.spec_from_file_location(
        "headroom.integrations.litellm_callback",
        __file__.replace(
            "tests/test_integrations/test_litellm_callback.py",
            "headroom/integrations/litellm_callback.py",
        ),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.HeadroomCallback  # type: ignore[attr-defined]


HeadroomCallback = _import_callback()


class TestHeadroomCallbackPostCallSuccessHook:
    """async_post_call_success_hook must exist and return response unchanged."""

    def test_method_exists(self) -> None:
        cb = HeadroomCallback()
        assert hasattr(cb, "async_post_call_success_hook"), (
            "HeadroomCallback must define async_post_call_success_hook "
            "for LiteLLM proxy compatibility"
        )

    def test_method_is_coroutine(self) -> None:
        cb = HeadroomCallback()
        assert inspect.iscoroutinefunction(cb.async_post_call_success_hook)

    @pytest.mark.asyncio
    async def test_returns_response_unchanged(self) -> None:
        cb = HeadroomCallback()
        sentinel = object()
        result = await cb.async_post_call_success_hook(
            data={},
            user_api_key_dict=None,
            response=sentinel,
        )
        assert result is sentinel
