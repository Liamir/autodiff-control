from pathlib import Path
import os
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42

PACKAGE_ROOT = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
PLOTS_DIR = PACKAGE_ROOT / "plots"


def save_fig(fig, name):
    """
    Save a matplotlib figure to the PLOTS_DIR as a PDF file.

    Args:
        fig: The matplotlib figure object to save.
        name: The filename (without extension) or with .pdf extension.
    """
    # Ensure the filename ends with .pdf
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    save_path = PLOTS_DIR / name
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to {save_path}")
