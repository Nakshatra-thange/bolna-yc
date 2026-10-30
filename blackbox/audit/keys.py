"""
Black Box - device keys.

  signing key  Ed25519. Signs every chain entry. The PUBLIC key is pinned at the server / verifier;
               the private key never leaves the device.
  audio key    AES-256-GCM. Encrypts voice clips at rest (a stolen device leaks no patient audio).

Hackathon storage: files with 0600 permissions in ~/.blackbox (or any folder you pass).
Production: generate the signing key INSIDE Secure Enclave / Android Keystore / TPM so it can
never be exported - that is what stops someone with the device from re-signing a forged history.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

RAW = serialization.Encoding.Raw


class KeyStore:
    def __init__(self, folder: str | Path | None = None):
        self.dir = Path(folder or Path.home() / ".blackbox")
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        self.signing_key = Ed25519PrivateKey.from_private_bytes(self._secret("signing.key"))
        self.audio_key = self._secret("audio.key")

    def _secret(self, name: str) -> bytes:
        p = self.dir / name
        if not p.exists():
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(os.urandom(32))
        return p.read_bytes()

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.signing_key.public_key()

    @property
    def public_key_hex(self) -> str:
        return self.public_key.public_bytes(RAW, serialization.PublicFormat.Raw).hex()

    @property
    def device_id(self) -> str:
        return "BB-" + hashlib.sha256(bytes.fromhex(self.public_key_hex)).hexdigest()[:12]

    def sign(self, data: bytes) -> bytes:
        return self.signing_key.sign(data)

    def export_public(self, path: str | Path) -> Path:
        """Write the public key for pinning at the server / in the verifier."""
        p = Path(path)
        p.write_text(self.public_key_hex + "\n")
        return p


def load_public_key(hex_or_path: str) -> Ed25519PublicKey:
    s = hex_or_path
    if Path(s).exists():
        s = Path(s).read_text().strip()
    return Ed25519PublicKey.from_public_bytes(bytes.fromhex(s))