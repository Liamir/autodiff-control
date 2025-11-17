import matplotlib as mpl
import seaborn as sns


def set_style():
    """Default style for matplotlib and seaborn plots."""
    sns.set_style("ticks")
    sns.color_palette(palette="plasma")
    mpl.rcParams["pdf.fonttype"] = 42  # Nature compliance
