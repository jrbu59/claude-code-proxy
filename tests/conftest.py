"""Pytest configuration for unit tests."""

import os


# Provide a default API key so importing config succeeds during tests.
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("BUFFER_STREAM_RATE_LIMIT_MS", "0")
os.environ.setdefault("BUFFER_STREAM_CACHE_TTL_SECONDS", "0")
