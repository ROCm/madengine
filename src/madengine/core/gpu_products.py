"""AMD Instinct product names and the compiler target (gfx) each one is.

One table for every place madengine turns a product label into an architecture (a user's
`--target-archs mi355x`, a Kubernetes node's product label). A product whose target is not yet
published maps to None rather than to a guess.
"""
from typing import Optional

# Longest prefix first: "MI300X" before "MI300".
PRODUCT_ARCH = (
    ("MI455", None),        # CDNA 5: no compiler target in this table yet
    ("MI355", "gfx950"),
    ("MI350", "gfx950"),
    ("MI325", "gfx942"),
    ("MI308", "gfx942"),
    ("MI300", "gfx942"),    # MI300X and MI300A (gfx940/gfx941 were pre-release targets)
    ("MI250", "gfx90a"),
    ("MI210", "gfx90a"),
    ("MI200", "gfx90a"),
    ("MI100", "gfx908"),
)


def arch_of_product(product: str) -> Optional[str]:
    """The gfx target of an AMD product label ("MI355X", "AMD Instinct MI300X", "mi-200"), or None."""
    name = (product or "").upper().replace("-", "").replace("_", "").replace(" ", "")
    for prefix, arch in PRODUCT_ARCH:
        if prefix in name:
            return arch
    return None


def known_product(product: str) -> bool:
    name = (product or "").upper().replace("-", "").replace("_", "").replace(" ", "")
    return any(prefix in name for prefix, _ in PRODUCT_ARCH)
