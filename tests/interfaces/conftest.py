"""Interface unit tests must never contact a real API endpoint."""
import socket
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Unit tests must mock network transport")
    monkeypatch.setattr(socket, "create_connection", reject)
