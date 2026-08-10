"""Convert the runner script into a notebook.

The pipeline lives in `notebooks/kaggle_memorization_run.py` so that git can show
diffs of it; Kaggle and Colab want .ipynb. Cells are split on `# %%` markers,
`# %% [markdown]` blocks become markdown cells.
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_cells(source: str):
    parts = re.split(r"^# %%(.*)$", source, flags=re.M)
    cells = []
    if parts[0].strip():
        cells.append(("code", parts[0]))
    for header, body in zip(parts[1::2], parts[2::2]):
        kind = "markdown" if "[markdown]" in header else "code"
        if kind == "markdown":
            body = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.strip().split("\n"))
        cells.append((kind, body.strip("\n")))
    return [c for c in cells if c[1].strip()]


def build(py_path: str, ipynb_path: str):
    source = open(py_path, encoding="utf-8").read()
    cells = []
    for kind, body in to_cells(source):
        lines = [l + "\n" for l in body.split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return len(cells)


if __name__ == "__main__":
    py = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "notebooks", "kaggle_memorization_run.py")
    ipynb = py.replace(".py", ".ipynb")
    n = build(py, ipynb)
    print(f"wrote {ipynb} ({n} cells)")
