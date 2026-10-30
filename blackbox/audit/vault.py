"""AES-256-GCM helpers for encrypting audio clips at rest."""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_NONCE_BYTES = 12


def encrypt_file(path: str | Path, key: bytes, aad: str) -> Path:
    """Encrypt *path* to ``<path>.enc`` and remove the plaintext on success."""
    source = Path(path)
    target = Path(f"{source}.enc")
    nonce = os.urandom(_NONCE_BYTES)
    encrypted = nonce + AESGCM(key).encrypt(nonce, source.read_bytes(), aad.encode("utf-8"))
    target.write_bytes(encrypted)
    os.chmod(target, 0o600)
    source.unlink()
    return target


def decrypt_file(path: str | Path, key: bytes, aad: str) -> bytes:
    """Authenticate and decrypt a file created by :func:`encrypt_file`."""
    payload = Path(path).read_bytes()
    if len(payload) < _NONCE_BYTES + 16:
        raise ValueError("encrypted audio is truncated")
    nonce, ciphertext = payload[:_NONCE_BYTES], payload[_NONCE_BYTES:]
    return AESGCM(key).decrypt(nonce, ciphertext, aad.encode("utf-8"))
