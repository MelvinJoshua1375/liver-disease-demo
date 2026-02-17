"""Shared color palette and matplotlib defaults for consistent visuals."""

import matplotlib as mpl

# Brand palette
PALETTE: dict[str, str] = {
    "primary":   "#4A90D9",   # blue
    "secondary": "#1A1A2E",   # dark navy
    "accent":    "#E86C00",   # orange
    "positive":  "#E86C00",   # orange for Positive class
    "negative":  "#4A90D9",   # blue for Negative class
    "neutral":   "#6C757D",   # grey
    "success":   "#4CAF50",   # green
    "warning":   "#FFC107",   # amber
}

# Model comparison palette (one color per model)
MODEL_COLORS: list[str] = [
    "#4A90D9",  # LR
    "#E86C00",  # DT
    "#4CAF50",  # SVC
    "#9C27B0",  # NB
    "#1A1A2E",  # RF
    "#F44336",  # DT+LR hybrid
]

# Global rcParams
mpl.rcParams.update({
    "figure.dpi":         120,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.2,
    "grid.linestyle":     "--",
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.7,
})
