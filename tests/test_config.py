"""
Tests for config.py's Razorpay key validation.

require_razorpay_keys() previously only checked that the two env vars were
non-empty -- nothing verified the key was actually a test-mode key. Pasting
a live key (rzp_live_...) into .env by mistake would authenticate against
Razorpay's real API with no warning anywhere, while the dashboard's top bar
still confidently displays "TEST MODE".
"""

import importlib

import pytest

import config


def _reload_config_with_keys(monkeypatch, key_id, key_secret="a-secret"):
    monkeypatch.setenv("RAZORPAY_KEY_ID", key_id or "")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", key_secret or "")
    importlib.reload(config)
    return config


def test_require_razorpay_keys_accepts_a_test_key(monkeypatch):
    cfg = _reload_config_with_keys(monkeypatch, "rzp_test_abc123")
    key_id, key_secret = cfg.require_razorpay_keys()
    assert key_id == "rzp_test_abc123"


def test_require_razorpay_keys_rejects_a_live_key(monkeypatch):
    cfg = _reload_config_with_keys(monkeypatch, "rzp_live_abc123")
    with pytest.raises(RuntimeError, match="test-mode"):
        cfg.require_razorpay_keys()


def test_require_razorpay_keys_rejects_an_unrecognized_key_format(monkeypatch):
    cfg = _reload_config_with_keys(monkeypatch, "some-other-key-format")
    with pytest.raises(RuntimeError, match="test-mode"):
        cfg.require_razorpay_keys()


def test_require_razorpay_keys_still_rejects_missing_keys(monkeypatch):
    cfg = _reload_config_with_keys(monkeypatch, "", "")
    with pytest.raises(RuntimeError, match="not set"):
        cfg.require_razorpay_keys()
