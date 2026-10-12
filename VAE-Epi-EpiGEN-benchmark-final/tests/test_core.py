from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script(rel):
    path = ROOT / rel
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_hits_target():
    m = load_script(Path("scripts") / "01_calibrate_epigen_alpha.py")
    rng = np.random.default_rng(1)
    G = {
        "rs1": rng.binomial(2, 0.35, size=1000).astype(float),
        "rs2": rng.binomial(2, 0.40, size=1000).astype(float),
    }
    pairs = pd.DataFrame({"SNP1": ["rs1"], "SNP2": ["rs2"]})
    alpha, frac, _ = m.find_alpha(
        G, pairs, "multiplicative", target=0.10, noise_sd=1.0,
        tolerance=1e-4, max_iter=100
    )
    assert alpha > 1
    assert abs(frac - 0.10) < 2e-4


def test_pair_key_order_invariant():
    m = load_script(Path("scripts") / "core" / "02_run_tree_ml_baselines.py")
    assert m.pair_key("rs2", "rs1") == m.pair_key("rs1", "rs2")


def test_prepare_canonical_pair_extracts_rsid():
    m = load_script(Path("scripts") / "core" / "00_prepare_sim1_inputs.py")
    assert m.canonical_pair("rs2:20:A:G", "rs1:10:C:T") == "rs1||rs2"
