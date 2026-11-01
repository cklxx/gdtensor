"""CRC must reject changed payloads, not compare an output with itself."""

import pytest

from tests.support.crc import crc_of, verify_payload


def test_crc_matches_standard_known_vector():
    assert crc_of(b"123456789") == "cbf43926"


@pytest.mark.parametrize("raw", [b"", b"\x00\xff\x91\x73\x0a", bytes(range(256))])
def test_crc_accepts_unchanged_payload(raw, capsys):
    expected = crc_of(raw)
    verify_payload("unchanged", expected, bytes(bytearray(raw)))
    assert "[CRC][VERIFY][PASS]" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["flip", "truncate", "swap", "append"])
def test_crc_rejects_corruption(damage, capsys):
    source = bytearray(b"\x91\x27\x00\xfe\x12\x55\xc3\x08")
    expected = crc_of(bytes(source))
    if damage == "flip":
        source[2] ^= 1
    elif damage == "truncate":
        del source[-1]
    elif damage == "swap":
        source[:4], source[4:] = source[4:], source[:4]
    else:
        source.append(0)
    with pytest.raises(AssertionError, match="damaged"):
        verify_payload("damaged", expected, bytes(source))
    assert "[PASS]" not in capsys.readouterr().out
