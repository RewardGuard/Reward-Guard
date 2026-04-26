"""
RewardGuard — AI Alignment and Reward Balance Analysis
=======================================================

Detect reward hacking, misalignment, and training degradation in
reinforcement learning systems.

Free Version — Analysis and Recommendations
--------------------------------------------
Primary API (live in-loop monitoring)::

    import rewardguard as rg

    monitor = rg.Monitor(
        expected={"food": 0.6, "survival": 0.4},
        tolerance=5.0,
        window=200,
    )

    for episode in range(num_episodes):
        for step in range(max_steps):
            r_food, r_survival = env.step(action)
            monitor.step({"food": r_food, "survival": r_survival})

    monitor.print_report()

Log-file API (post-hoc analysis)::

    import rewardguard as rg

    episodes = rg.parse_logs(open("training.log").read())
    result   = rg.analyze_balance(episodes, expected={"food": 60, "survival": 40})
    rg.RewardGuard().print_analysis_report(result)

For auto-correction and advanced statistical detection upgrade to
RewardGuard Premium: https://rewardguard.dev/premium
"""

__version__ = "1.0.0"
__author__ = "RewardGuard Team"
__email__ = "giovan@rewardguard.dev"
__license__ = "MIT"

from .Core import (
    # Primary in-loop API
    Monitor,
    # Data structures
    EpisodeData,
    AnalysisResult,
    # Classes
    LogParser,
    RewardAnalyzer,
    RewardGuard,
    # Convenience functions
    parse_logs,
    analyze_balance,
    recommend_weights,
)
from .plot import plot_balance

__all__ = [
    "__version__",
    # Primary API
    "Monitor",
    # Data structures
    "EpisodeData",
    "AnalysisResult",
    # Classes
    "LogParser",
    "RewardAnalyzer",
    "RewardGuard",
    # Convenience functions
    "parse_logs",
    "analyze_balance",
    "recommend_weights",
    # Visualization
    "plot_balance",
]
