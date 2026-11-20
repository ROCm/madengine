"""madengine.core.gpu_products: one product -> compiler target table."""
from madengine.core.gpu_products import arch_of_product
from madengine.deployment.k8s_results import gpu_arch_from_product_name
from madengine.execution.dockerfile_utils import normalize_architecture_name


def test_current_and_next_generation_products():
    assert arch_of_product("AMD Instinct MI355X") == "gfx950"
    assert arch_of_product("MI350X") == "gfx950"
    assert arch_of_product("MI325X") == "gfx942"
    assert arch_of_product("mi-300x") == "gfx942"
    assert arch_of_product("MI455X") is None          # no target in the table: no guess
    assert arch_of_product("Radeon") is None


def test_both_callers_use_it():
    assert normalize_architecture_name("mi355x") == "gfx950"
    assert normalize_architecture_name("mi300a") == "gfx942"
    assert normalize_architecture_name("mi455x") == "mi455x"
    assert gpu_arch_from_product_name("AMD-Instinct-MI355X") == "gfx950"
