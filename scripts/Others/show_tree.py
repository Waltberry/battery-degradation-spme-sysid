# scripts/show_tree.py
import os

EXCLUDE = {".venv", "__pycache__", ".git"}

for root, dirs, files in os.walk(".", topdown=True):
    dirs[:] = [d for d in dirs if d not in EXCLUDE]

    level = root.count(os.sep)
    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")

    subindent = " " * 4 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")