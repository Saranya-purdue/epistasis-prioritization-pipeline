from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = [
    "/" + "scratch/",
    "/" + "home/",
    "#" + "SBATCH",
    r"\b" + "S" + "LURM" + r"\b",
]


def test_no_environment_specific_paths_or_scheduler_directives():
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".r", ".md", ".txt", ".yml", ".yaml"}:
            continue
        text = path.read_text(errors="ignore")
        for pattern in FORBIDDEN:
            if re.search(pattern, text, flags=re.I):
                hits.append((str(path.relative_to(ROOT)), pattern))
    assert hits == []
