"""
plot_style.py
=============
Shared figure style for AOAS revision plots.
Toggle PRINT_MODE to switch between color (online) and grayscale (print).

Usage:
    from plot_style import STYLE, COLORS, MARKERS, LABELS, apply_style
    apply_style()
"""

import matplotlib.pyplot as plt

# ============================================================
# Toggle: False = color (online), True = grayscale (print)
# ============================================================
PRINT_MODE = False

# ============================================================
# Palette
# ============================================================
if PRINT_MODE:
    COLORS = {
        'TG_SSP': '#AAAAAA',
        'IBP': '#777777',
        'NB_SSP_regression': '#333333',
        'NB_SSP_MLE': '#555555',
        'exp_0': '#333333',
        'exp_1': '#777777',
        'exp_2': '#AAAAAA',
    }
else:
    COLORS = {
        'TG_SSP': '#e41a1c',       # red
        'IBP': '#377eb8',           # blue
        'NB_SSP_regression': '#4daf4a',  # green
        'NB_SSP_MLE': '#984ea3',    # purple
        'exp_0': '#1b9e77',         # teal
        'exp_1': '#d95f02',         # orange
        'exp_2': '#7570b3',         # slate
    }

# Always use distinct markers + line styles (readable in both modes)
MARKERS = {
    'TG_SSP': 'o',
    'IBP': 's',
    'NB_SSP_regression': '^',
    'NB_SSP_MLE': 'D',
    'exp_0': 'o',
    'exp_1': 's',
    'exp_2': '^',
}

LINESTYLES = {
    'TG_SSP': '--',
    'IBP': '-.',
    'NB_SSP_regression': '-',
    'NB_SSP_MLE': ':',
    'exp_0': '-',
    'exp_1': '--',
    'exp_2': '-.',
}

LABELS = {
    'TG_SSP': 'TG-SSP',
    'IBP': 'IBP',
    'NB_SSP_regression': 'NB-SSP (reg)',
    'NB_SSP_MLE': 'NB-SSP (MLE)',
}


def apply_style():
    """Apply shared rcParams."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.figsize': (6, 4),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
