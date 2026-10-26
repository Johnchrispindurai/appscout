"""Turn the notebook sources in notebooks/src/*.py into executed .ipynb files.

A source file is plain Python split into cells by `# %%` (code) and `# %% [markdown]`
(text, every line starting with `# `). Keeping the sources as .py files keeps them
readable in a diff; the .ipynb files are what gets opened in Jupyter or Colab.

    python notebooks/build.py              # all
    python notebooks/build.py 03           # only sources whose name starts with 03
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent


def parse(text: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    kind, lines = None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(lines).strip("\n")
        if kind == "md":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
            nb.cells.append(nbformat.v4.new_markdown_cell(body))
        elif body:
            nb.cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("md" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    flush()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for src in sorted((HERE / "src").glob(f"{only}*.py")):
        nb = parse(src.read_text(encoding="utf-8"))
        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(HERE)}}).execute()
        out = HERE / f"{src.stem}.ipynb"
        nbformat.write(nb, out)
        print("built", out.name, flush=True)


if __name__ == "__main__":
    main()
