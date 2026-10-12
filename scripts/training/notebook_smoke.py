"""
notebook_smoke.py — runs every notebook in notebooks/ with SMOKE=1 on the
CPU (a few seconds each): executes the code cells in order in one
namespace, from the repository root, with checkpoints and results in a
temporary directory, then checks the results file each one writes.

    python scripts/training/notebook_smoke.py [notebooks/07_*.ipynb ...]

CI runs it in the training-reference job (no network: SMOKE uses the
site's corpus). Exit status 1 if any notebook fails.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def smoke(path: Path) -> dict:
    nb = json.loads(path.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SMOKE"] = "1"
        os.environ["RUNS_DIR"] = tmp
        ns: dict = {"__name__": "__main__"}
        t0 = time.time()
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            exec(compile("".join(c["source"]), f"{path.name}", "exec"), ns)
        written = list(Path(tmp).glob("*.json"))
        if len(written) != 1:
            raise RuntimeError(f"expected one results file, found {written}")
        res = json.loads(written[0].read_text())
        assert res["metadata"]["smoke"] is True and res["rows"], "results file incomplete"
        return {"notebook": path.name, "seconds": round(time.time() - t0, 1),
                "results": written[0].name, "rows": len(res["rows"])}


def main() -> None:
    os.chdir(ROOT)
    paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    bad = 0
    for p in paths:
        try:
            print(json.dumps(smoke(p)))
        except Exception as e:  # report every notebook, then fail
            bad += 1
            print(f"FAILED {p.name}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
