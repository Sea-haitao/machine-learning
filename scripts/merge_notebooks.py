"""Merge all numbered phase notebooks into a single final.ipynb."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
OUTPUT = Path("final.ipynb")


def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge():
    notebooks = sorted(NOTEBOOKS_DIR.glob("[0-9]*.ipynb"))
    if not notebooks:
        print("No numbered notebooks found in notebooks/")
        return

    print(f"Merging {len(notebooks)} notebooks:")
    for nb in notebooks:
        print(f"  - {nb.name}")

    # Use the first notebook as the base (preserves kernel/metadata)
    merged = load_notebook(notebooks[0])
    merged["cells"] = list(merged["cells"])

    for nb_path in notebooks[1:]:
        nb = load_notebook(nb_path)
        # Add a markdown separator between sections
        separator = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"---\n", f"# {nb_path.stem}\n"],
        }
        merged["cells"].append(separator)
        merged["cells"].extend(nb["cells"])

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)

    print(f"\nCreated {OUTPUT} with {len(merged['cells'])} cells.")


if __name__ == "__main__":
    merge()
