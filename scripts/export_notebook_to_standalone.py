# export_notebook_to_standalone
#!/usr/bin/env python3
"""
Export the nonlinear synthetic validation notebook into a standalone Python script.

Why this script exists:
    - The notebook already contains the full workflow.
    - Manually rewriting every notebook cell into a separate script is risky.
    - This exporter preserves cell order and code while adding batch-safe behavior.

Output:
    scripts/run_nonlinear_synthetic_validation_full.py

Run from project root:
    python scripts/export_notebook_to_standalone.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_DIR = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")
NOTEBOOK_PATH = PROJECT_DIR / "notebooks" / "nonlinear_synthetic_validation_thesis_clean.ipynb"
OUT_SCRIPT_PATH = PROJECT_DIR / "scripts" / "run_nonlinear_synthetic_validation_full.py"


def clean_cell_source(source: str) -> str:
    """
    Make notebook code safer for non-interactive Slurm execution.
    """
    s = source

    # Remove notebook-level future imports.
    # The generated standalone script already has this at the very top.
    # Any later future import will crash Python.
    s = re.sub(
        r"^\s*from\s+__future__\s+import\s+annotations\s*\n",
        "",
        s,
        flags=re.MULTILINE,
    )

    # Remove IPython display import/use problems only where needed.
    # Keep display() as a harmless print-like fallback via prelude.
    s = s.replace(
        "from IPython.display import display",
        "# from IPython.display import display  # replaced by batch-safe display",
    )

    # Make real data path project-root-safe.
    # Notebook ran inside notebooks/, but Slurm will run from project root.
    s = re.sub(
        r'MPR_PATH\s*=\s*["\']12to1-25%CNC-3%GQDs _C01\.mpr["\']',
        'MPR_PATH = str(PROJECT_DIR / "data" / "12to1-25%CNC-3%GQDs _C01.mpr")',
        s,
    )

    return s.rstrip() + "\n"


def main() -> int:
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    if not code_cells:
        raise RuntimeError(f"No code cells found in notebook: {NOTEBOOK_PATH}")

    OUT_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prelude = f'''#!/usr/bin/env python3
# ============================================================
# AUTO-GENERATED STANDALONE SCRIPT
# Source notebook:
#   {NOTEBOOK_PATH}
#
# Do not edit this generated script manually unless needed.
# Regenerate with:
#   python scripts/export_notebook_to_standalone.py
# ============================================================

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path("{PROJECT_DIR}")
os.chdir(PROJECT_DIR)

# Make package imports work from project root.
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Non-interactive plotting for Slurm.
os.environ.setdefault("MPLBACKEND", "Agg")

# Thread settings. Slurm can override SLURM_CPUS_PER_TASK.
N_THREADS_EXPORT = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("N_THREADS", "8")))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS_EXPORT)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS_EXPORT)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS_EXPORT)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS_EXPORT)
os.environ["XLA_FLAGS"] = (
    f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={{N_THREADS_EXPORT}}"
)

# Create output directories.
for _p in [
    PROJECT_DIR / "results" / "logs",
    PROJECT_DIR / "results" / "figures",
    PROJECT_DIR / "results" / "figures" / "nonlinear_synthetic",
    PROJECT_DIR / "results" / "metrics",
    PROJECT_DIR / "results" / "tables",
    PROJECT_DIR / "results" / "outputs",
]:
    _p.mkdir(parents=True, exist_ok=True)

# Batch-safe display replacement.
def display(obj=None, *args, **kwargs):
    try:
        print(obj)
    except Exception:
        print(repr(obj))

# Batch-safe matplotlib setup.
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

_FIG_COUNTER = {{"n": 0}}
_FIG_DIR = PROJECT_DIR / "results" / "figures" / "nonlinear_synthetic"

def _batch_show(*args, **kwargs):
    """
    Replace plt.show() in notebook-exported code.
    Saves each open figure instead of trying to display interactively.
    """
    figs = list(map(plt.figure, plt.get_fignums()))
    if not figs:
        return

    for fig in figs:
        _FIG_COUNTER["n"] += 1
        out = _FIG_DIR / f"notebook_fig_{{_FIG_COUNTER['n']:04d}}.png"
        try:
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[saved figure] {{out}}")
        except Exception as exc:
            print(f"[warn] could not save figure {{out}}: {{exc}}")
    plt.close("all")

plt.show = _batch_show

print("=" * 72)
print("STANDALONE NONLINEAR SYNTHETIC VALIDATION SCRIPT")
print("=" * 72)
print("Project dir:", PROJECT_DIR)
print("Source notebook:", Path("{NOTEBOOK_PATH}"))
print("Python:", sys.executable)
print("Threads:", N_THREADS_EXPORT)
print("=" * 72)

'''

    parts = [prelude]

    for idx, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        parts.append("\n\n")
        parts.append("# " + "=" * 72 + "\n")
        parts.append(f"# EXPORTED NOTEBOOK CODE CELL {idx}\n")
        parts.append("# " + "=" * 72 + "\n")
        parts.append(clean_cell_source(source))

    postlude = '''

print("=" * 72)
print("STANDALONE SCRIPT COMPLETE")
print("Figures saved under:", PROJECT_DIR / "results" / "figures" / "nonlinear_synthetic")
print("=" * 72)
'''

    parts.append(postlude)

    OUT_SCRIPT_PATH.write_text("".join(parts), encoding="utf-8")
    OUT_SCRIPT_PATH.chmod(0o755)

    print("=" * 72)
    print("Export complete")
    print("Input notebook:", NOTEBOOK_PATH)
    print("Output script :", OUT_SCRIPT_PATH)
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())