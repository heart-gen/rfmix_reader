"""Reader utilities for RFMix and related formats."""

from __future__ import annotations

__all__ = [
    "read_fb",
    "read_flare",
    "read_rfmix",
    "read_rfmix_fb",
    "read_simu",
    "extract_locus_ancestry",
]

_lazy = {
    "read_fb": (".fb_read", "read_fb"),
    "read_flare": (".read_flare", "read_flare"),
    "read_rfmix": (".read_msp", "read_rfmix"),
    "extract_locus_ancestry": (".read_msp", "extract_locus_ancestry"),
    "read_rfmix_fb": (".read_rfmix", "read_rfmix_fb"),
    "read_simu": (".read_simu", "read_simu"),
}


def __getattr__(name: str):
    if name in _lazy:
        import importlib

        mod_name, attr_name = _lazy[name]
        mod = importlib.import_module(mod_name, __name__)
        obj = getattr(mod, attr_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
