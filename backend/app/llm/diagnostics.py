"""Allowlisted exception metadata only; never stringify requests or exceptions."""
import errno
import socket
import ssl
from http.client import RemoteDisconnected


def connection_diagnostics(error):
    chain, seen = [], set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    classes = [type(item).__name__ for item in chain]
    if any(isinstance(item, TimeoutError) or "timeout" in type(item).__name__.lower() for item in chain):
        category = "timeout"
    elif any(isinstance(item, socket.gaierror) for item in chain):
        category = "DNS"
    elif any(isinstance(item, ssl.SSLError) for item in chain):
        category = "TLS"
    elif any(isinstance(item, RemoteDisconnected) for item in chain):
        category = "remote disconnect"
    elif any(isinstance(item, ConnectionResetError) or getattr(item, "errno", None) == errno.ECONNRESET for item in chain):
        category = "connection reset"
    else:
        category = "unknown"
    return {"exception_class": classes[0], "underlying_cause_classes": classes[1:],
            "category": category, "timeout": "YES" if category == "timeout" else "UNKNOWN" if category == "unknown" else "NO"}
