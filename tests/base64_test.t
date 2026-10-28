# tests/base64_test.t — Kiểm thử module std/base64

let b64 = load("std/base64")

fn test_rfc4648_test_vectors():
    print("=== Test 1: RFC 4648 Standard Vectors ===")
    assert b64.b64encode("") == ""
    assert b64.b64decode("") == ""

    assert b64.b64encode("f") == "Zg=="
    assert b64.b64decode("Zg==") == "f"

    assert b64.b64encode("fo") == "Zm8="
    assert b64.b64decode("Zm8=") == "fo"

    assert b64.b64encode("foo") == "Zm9v"
    assert b64.b64decode("Zm9v") == "foo"

    assert b64.b64encode("foob") == "Zm9vYg=="
    assert b64.b64decode("Zm9vYg==") == "foob"

    assert b64.b64encode("fooba") == "Zm9vYmE="
    assert b64.b64decode("Zm9vYmE=") == "fooba"

    assert b64.b64encode("foobar") == "Zm9vYmFy"
    assert b64.b64decode("Zm9vYmFy") == "foobar"

    print("RFC 4648 Vectors: PASSED")

fn test_roundtrip_long_text():
    print("=== Test 2: Long Text Roundtrip ===")
    let original = "Ngon ngu lap trinh T - Nhanh, nhe, doc lap, thich hop cho AI va Transformer inference!"
    let encoded = b64.b64encode(original)
    let decoded = b64.b64decode(encoded)
    assert decoded == original
    print("Encoded length:", len(encoded))
    print("Decoded matches original: PASSED")

fn test_binary_roundtrip():
    print("=== Test 3: Binary Bytes 0..255 Roundtrip ===")
    mut raw_chars = []
    for i in range(256):
        raw_chars.append(chr(i))
    let binary_str = "".join(raw_chars)
    assert len(binary_str) == 256

    let enc = b64.b64encode(binary_str)
    let dec = b64.b64decode(enc)
    assert dec == binary_str
    assert len(dec) == 256
    print("Binary Roundtrip 0..255: PASSED")

test_rfc4648_test_vectors()
test_roundtrip_long_text()
test_binary_roundtrip()
print("All base64 tests passed successfully!")
