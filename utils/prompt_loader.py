import os


def _resolve_prompt_basename(filename: str) -> str:
    """Return `name.md` under prompts/ (strips path; .txt and .md both map to the same file)."""
    base = os.path.basename(filename)
    root, ext = os.path.splitext(base)
    ext_lower = ext.lower()
    if ext_lower in (".md", ".txt"):
        stem = root
    elif ext_lower == "":
        stem = base
    else:
        stem = root
    return f"{stem}.md"


def load_prompt(filename: str) -> str:
    """Load a Markdown prompt from the prompts/ directory."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_path, "prompts", _resolve_prompt_basename(filename))

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
