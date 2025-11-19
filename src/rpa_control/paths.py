from pathlib import Path
import os

PACKAGE_ROOT = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent

PLOTS_DIR = PACKAGE_ROOT / "plots"

# make directories if they don't exist:
PLOTS_DIR.mkdir(parents=False, exist_ok=True)


def save_fig(fig, name):
    """
    Save a matplotlib figure to the PLOTS_DIR as an SVG file.

    Args:
        fig: The matplotlib figure object to save.
        name: The filename (without extension) or with .svg extension.
    """
    # .pdf is Nature compliant. Allows editing text post-hoc.
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    save_path = PLOTS_DIR / name
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {save_path}")
