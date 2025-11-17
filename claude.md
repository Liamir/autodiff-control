# How I like to work with you (claude):

Please implement small readable functions, and let me approve them one by one.
Lots of asserts and tests.

# Linear MCP:

Always show both the id and description of the issue. I don't remember ids by themselves.

# Plot style:

sns.despine(ax=ax)
default figsize((6, 3))
legend(frameon=False, bbox_to_anchor=(1, 1))
lowercase axis labels

# Jupyter notebooks:

Always start with:
%load_ext autoreload
%autoreload 2

# Git

Don't add co-authored by claude.