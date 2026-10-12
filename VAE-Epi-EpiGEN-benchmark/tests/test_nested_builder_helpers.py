import importlib.util
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'benchmark_b'/'01_build_nested_benchmark_genotypes.py'
spec=importlib.util.spec_from_file_location('nested',SCRIPT)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_rsid_extraction():
    assert mod.rsid('1:10:rs123:A:G')=='rs123'
    assert mod.rsid('rs99')=='rs99'
    assert mod.rsid('1:10:A:G')=='1:10:A:G'
