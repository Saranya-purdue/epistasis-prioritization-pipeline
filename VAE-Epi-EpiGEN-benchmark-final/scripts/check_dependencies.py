#!/usr/bin/env python3
"""Report availability of optional dependencies used by the benchmark."""

import argparse
from pathlib import Path
import importlib.util
import shutil
import sys


def module_status(name):
    return importlib.util.find_spec(name) is not None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epigen-root", default=None)
    p.add_argument("--plink", default="plink")
    args = p.parse_args()

    print("Python:", sys.version.split()[0])
    for pkg in ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "shap", "tensorflow"]:
        print(f"{pkg:12s}:", "OK" if module_status(pkg) else "not installed")

    print("PLINK:", shutil.which(args.plink) or "not found")

    if args.epigen_root:
        root = Path(args.epigen_root).expanduser().resolve()
        expected = root / "utils" / "parametrized_model.py"
        print("EpiGEN:", "OK" if expected.exists() else f"not found at {expected}")
    else:
        print("EpiGEN: not checked (pass --epigen-root)")


if __name__ == "__main__":
    main()
