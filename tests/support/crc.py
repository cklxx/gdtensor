"""Independent CRC checks for non-arange test payloads.

Record the expected CRC before the tested operation, or from an independent
output oracle. Compare it with the resulting bytes. Retain exact-byte and
numeric assertions: CRC does not replace either check. Arange tests keep
their existing element or byte comparisons without additional CRC checks.
"""

import zlib


def crc_of(raw: bytes) -> str:
    return f"{zlib.crc32(raw):08x}"


def verify_payload(label: str, want_crc: str, raw: bytes) -> None:
    got = crc_of(raw)
    assert got == want_crc, (label, want_crc, got)
    print(
        f"[CRC][VERIFY][PASS] checks=payload_crc label={label} CRC={got}",
        flush=True,
    )
