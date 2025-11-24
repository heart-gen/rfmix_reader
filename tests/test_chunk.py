import pytest
from rfmix_reader.chunk import Chunk

def test_default_chunk_values():
    """Check default values of Chunk."""
    c = Chunk()
    assert c.nsamples == 1024
    assert c.nloci == 1024

def test_override_chunk_values():
    """Check overriding defaults with custom values."""
    c = Chunk(nsamples=512, nloci=2048)
    assert c.nsamples == 512
    assert c.nloci == 2048

def test_none_values():
    """Check None values are allowed (include all samples/loci)."""
    c = Chunk(nsamples=None, nloci=None)
    assert c.nsamples is None
    assert c.nloci is None

def test_partial_none_values():
    """Check setting only one parameter to None works correctly."""
    c1 = Chunk(nsamples=None, nloci=128)
    c2 = Chunk(nsamples=256, nloci=None)
    assert c1.nsamples is None and c1.nloci == 128
    assert c2.nsamples == 256 and c2.nloci is None

def test_equality_between_chunks():
    """Dataclass equality should work as expected."""
    c1 = Chunk(256, 128)
    c2 = Chunk(256, 128)
    c3 = Chunk(512, 128)
    assert c1 == c2
    assert c1 != c3

def test_repr_and_str_snapshot():
    """Check that repr and str give informative output."""
    c = Chunk(100, 200)
    out = repr(c)
    assert "Chunk" in out
    assert "nsamples=100" in out
    assert "nloci=200" in out
