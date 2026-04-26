"""
RewardGuard — Balance visualization.

Requires matplotlib::

    pip install matplotlib

Usage::

    import rewardguard as rg
    from rewardguard.plot import plot_balance

    monitor = rg.Monitor(expected={"task": 0.7, "safety": 0.3})
    # ... training loop ...
    plot_balance(monitor.check())
    # or save to file:
    plot_balance(monitor.check(), save_path="balance_report.png")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .Core import AnalysisResult


_SEVERITY_COLORS = {
    "ok": "#2ecc71",       # green
    "warning": "#f39c12",  # orange
    "critical": "#e74c3c", # red
}
_EXPECTED_COLOR = "#3498db"  # blue


def plot_balance(
    result: "AnalysisResult",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot a two-panel balance report for an :class:`~rewardguard.AnalysisResult`.

    Left panel  — Horizontal grouped bar chart: real vs expected % per component,
                  bars colored by severity (green / orange / red).
    Right panel — Suggested weight multipliers, with a dashed line at 1.0 (no change).

    Args:
        result:     Output of :meth:`~rewardguard.Monitor.check` or
                    :func:`~rewardguard.analyze_balance`.
        title:      Optional chart title.
        save_path:  If provided, save the figure to this path (e.g. ``"report.png"``).
        show:       Call ``plt.show()`` after rendering (default True).
                    Set to False when saving headlessly.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_balance. "
            "Install it with: pip install matplotlib"
        ) from exc

    sources = result.sources_found
    n = len(sources)
    if n == 0:
        raise ValueError("No sources in AnalysisResult — nothing to plot.")

    real = [result.real_percentages.get(s, 0.0) for s in sources]
    expected = [result.expected_percentages.get(s, 0.0) for s in sources]
    severities = [result.imbalance_report[s]["severity"] for s in sources]
    bar_colors = [_SEVERITY_COLORS[sev] for sev in severities]
    weights = [result.suggested_reward_weights.get(s, 1.0) for s in sources]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4, n * 0.9 + 2)))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#eeeeee")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    y = np.arange(n)
    bar_h = 0.35

    # --- Left panel: real vs expected ---
    bars_real = ax1.barh(y + bar_h / 2, real, bar_h, color=bar_colors, alpha=0.9, label="Real %")
    bars_exp = ax1.barh(y - bar_h / 2, expected, bar_h, color=_EXPECTED_COLOR, alpha=0.5, label="Expected %")

    ax1.set_yticks(y)
    ax1.set_yticklabels(sources, color="#dddddd")
    ax1.set_xlabel("Percentage (%)", color="#cccccc")
    ax1.set_title("Reward Component Balance", color="#eeeeee", fontsize=12, pad=10)
    ax1.axvline(0, color="#555577", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars_real, real):
        ax1.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", color="#eeeeee", fontsize=8,
        )

    # Legend + severity legend
    legend_patches = [
        mpatches.Patch(color=_EXPECTED_COLOR, alpha=0.5, label="Expected"),
        mpatches.Patch(color=_SEVERITY_COLORS["ok"], label="OK"),
        mpatches.Patch(color=_SEVERITY_COLORS["warning"], label="Warning"),
        mpatches.Patch(color=_SEVERITY_COLORS["critical"], label="Critical"),
    ]
    ax1.legend(handles=legend_patches, loc="lower right",
               facecolor="#22224a", edgecolor="#555577", labelcolor="#cccccc", fontsize=8)

    # Unexpected source annotation
    for i, s in enumerate(sources):
        if s in result.unexpected_sources:
            ax1.text(
                0.5, y[i], " ⚠ unexpected", va="center", ha="left",
                color="#f39c12", fontsize=7, style="italic",
            )

    # --- Right panel: weight multipliers ---
    weight_colors = [
        _SEVERITY_COLORS["critical"] if w >= 4.5 or w <= 0.15
        else _SEVERITY_COLORS["warning"] if w >= 2.0 or w <= 0.5
        else _SEVERITY_COLORS["ok"]
        for w in weights
    ]
    bars_w = ax2.barh(y, weights, 0.5, color=weight_colors, alpha=0.85)
    ax2.axvline(1.0, color="#aaaacc", linewidth=1.2, linestyle="--", label="No change (1.0×)")
    ax2.set_yticks(y)
    ax2.set_yticklabels(sources, color="#dddddd")
    ax2.set_xlabel("Weight multiplier", color="#cccccc")
    ax2.set_title("Suggested Weight Adjustments", color="#eeeeee", fontsize=12, pad=10)

    for bar, val in zip(bars_w, weights):
        ax2.text(
            bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}×", va="center", ha="left", color="#eeeeee", fontsize=8,
        )

    ax2.legend(facecolor="#22224a", edgecolor="#555577", labelcolor="#cccccc", fontsize=8)

    # --- Overall title ---
    severity_label = {
        "ok": "✓ All balanced",
        "warning": "⚠ Warnings detected",
        "critical": "✗ Critical imbalance",
    }.get(result.severity, result.severity)
    chart_title = title or f"RewardGuard Report — {result.episode_count} steps — {severity_label}"
    fig.suptitle(chart_title, color="#ffffff", fontsize=13, y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
