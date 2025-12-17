from pathlib import Path
import os

PACKAGE_ROOT = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent

PLOTS_DIR = PACKAGE_ROOT / "plots"

# make directories if they don't exist:
PLOTS_DIR.mkdir(parents=False, exist_ok=True)


def save_fig(fig, name):
    r"""
    Save a matplotlib figure as a PDF file.

    Args:
        fig: The matplotlib figure object to save.
        name: The filename (without extension) or full path.
              If it's a simple name, saves to PLOTS_DIR.
              If it's a path (contains / or \), uses it directly.
    """
    # .pdf is Nature compliant. Allows editing text post-hoc.
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"

    # Check if it's a full path (contains directory separators)
    name_path = Path(name)
    if name_path.is_absolute() or "/" in str(name) or "\\" in str(name):
        # Use the provided path directly
        save_path = name_path
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Simple filename - save to PLOTS_DIR
        save_path = PLOTS_DIR / name

    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {save_path}")
