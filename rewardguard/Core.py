#!/usr/bin/env python3
"""
RewardGuard - Reward Balance Analyzer
======================================
Analyzes RL training logs and detects reward component imbalances,
including reward hacking and component starvation.
"""

import itertools
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeData:
    """Individual episode data."""
    episode_num: int
    status: str
    total_reward: float
    reward_sources: Dict[str, float]
    percentages: Dict[str, float]

    def __post_init__(self) -> None:
        if self.percentages:
            total = sum(self.percentages.values())
            if total > 0:
                for key in self.percentages:
                    self.percentages[key] = round(
                        self.percentages[key] / total * 100, 4
                    )


@dataclass
class AnalysisResult:
    """Balance analysis result."""
    real_percentages: Dict[str, float]
    expected_percentages: Dict[str, float]
    imbalance_report: Dict[str, Dict[str, Any]]
    suggested_reward_weights: Dict[str, float]
    episode_count: int
    sources_found: List[str]
    # Overall severity across all components: "ok", "warning", "critical"
    severity: str = "ok"
    # Components present in real data but absent from expected distribution
    unexpected_sources: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_finite(value: Any, label: str) -> None:
    """Raise ValueError if value is not a finite number."""
    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        raise ValueError(
            f"{label} = {value!r} is not a finite number. "
            "All reward values must be real, finite floats."
        )


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

class LogParser:
    """Training log parser."""

    EPISODE_PATTERN = r'Ep\s+(\d+)\s*\|\s*(\w+)\s*\|\s*reward=([-\d.]+)'
    TOTAL_REWARD_PATTERN = r'total_reward:\s*([-\d.]+)'
    KV_PATTERN = r'^(\w+):\s*([-\d.]+)$'

    def parse_episode_block(self, lines: List[str]) -> Optional[EpisodeData]:
        """Parse a block of lines corresponding to a single episode."""
        episode_num: Optional[int] = None
        status: Optional[str] = None
        total_reward: Optional[float] = None
        reward_sources: Dict[str, float] = {}
        percentages: Dict[str, float] = {}
        parsing_section: Optional[str] = None

        for line in lines:
            line = line.strip()

            episode_match = re.match(self.EPISODE_PATTERN, line)
            if episode_match:
                episode_num = int(episode_match.group(1))
                status = episode_match.group(2)
                total_reward = float(episode_match.group(3))
                continue

            if line.startswith("=== EPISODE DATA ==="):
                parsing_section = "data"
                continue
            elif line.startswith("sources:"):
                parsing_section = "sources"
                continue
            elif line.startswith("percentages:"):
                parsing_section = "percentages"
                continue
            elif line == "":
                parsing_section = None
                continue

            if parsing_section == "data":
                m = re.match(self.TOTAL_REWARD_PATTERN, line)
                if m:
                    total_reward = float(m.group(1))

            elif parsing_section == "sources":
                m = re.match(self.KV_PATTERN, line)
                if m:
                    reward_sources[m.group(1)] = float(m.group(2))

            elif parsing_section == "percentages":
                m = re.match(self.KV_PATTERN, line)
                if m:
                    percentages[m.group(1)] = float(m.group(2))

        if episode_num is None or not reward_sources:
            return None

        if total_reward is None:
            total_reward = sum(reward_sources.values())

        return EpisodeData(
            episode_num=episode_num,
            status=status,
            total_reward=total_reward,
            reward_sources=reward_sources,
            percentages=percentages,
        )

    def parse_logs(self, raw_text: str) -> List[EpisodeData]:
        """Parse logs from multiple episodes."""
        episodes: List[EpisodeData] = []
        lines = raw_text.strip().split('\n')
        current_block: List[str] = []

        for line in lines:
            line = line.strip()
            if re.match(self.EPISODE_PATTERN, line) and current_block:
                episode = self.parse_episode_block(current_block)
                if episode:
                    episodes.append(episode)
                current_block = []
            current_block.append(line)

        if current_block:
            episode = self.parse_episode_block(current_block)
            if episode:
                episodes.append(episode)

        return episodes


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class RewardAnalyzer:
    """Reward balance analyzer."""

    def __init__(self, tolerance: float = 5.0) -> None:
        """
        Args:
            tolerance: Percentage-point tolerance for flagging imbalance (±5 pp).
        """
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self.tolerance = tolerance

    def aggregate_percentages(
        self, episodes: List[EpisodeData]
    ) -> Dict[str, float]:
        """Aggregate component percentages across episodes by reward magnitude."""
        if not episodes:
            return {}

        source_totals: Dict[str, float] = defaultdict(float)
        grand_total = 0.0

        for episode in episodes:
            for source, value in episode.reward_sources.items():
                mag = abs(value)
                source_totals[source] += mag
                grand_total += mag

        if grand_total == 0:
            return {s: 0.0 for s in source_totals}

        return {
            source: round(total / grand_total * 100, 2)
            for source, total in source_totals.items()
        }

    def detect_imbalance(
        self,
        real_percentages: Dict[str, float],
        expected_percentages: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect imbalances between real and expected percentage distributions.

        Severity levels:
        - ``"ok"``       — within tolerance
        - ``"warning"``  — between 1× and 3× tolerance
        - ``"critical"`` — more than 3× tolerance, or component completely absent
        """
        report: Dict[str, Dict[str, Any]] = {}
        all_sources = set(real_percentages) | set(expected_percentages)

        for source in all_sources:
            real_pct = real_percentages.get(source, 0.0)
            expected_pct = expected_percentages.get(source, 0.0)
            difference = real_pct - expected_pct
            abs_diff = abs(difference)

            # --- Severity classification ---
            if abs_diff <= self.tolerance:
                status = "balanced"
                severity = "ok"
                recommendation = "No action needed"
            elif abs_diff <= self.tolerance * 3:
                status = "imbalanced"
                severity = "warning"
                if difference > 0:
                    recommendation = f"Decrease weight by ~{abs_diff:.1f}%"
                else:
                    recommendation = f"Increase weight by ~{abs_diff:.1f}%"
            else:
                status = "imbalanced"
                severity = "critical"
                if difference > 0:
                    recommendation = f"CRITICAL: Decrease weight by ~{abs_diff:.1f}%"
                else:
                    recommendation = f"CRITICAL: Increase weight by ~{abs_diff:.1f}%"

            # --- Reward hacking: component completely absent ---
            if real_pct == 0.0 and expected_pct > self.tolerance:
                status = "imbalanced"
                severity = "critical"
                recommendation = (
                    f"REWARD HACKING DETECTED: '{source}' is completely absent "
                    f"(expected {expected_pct:.1f}%). Agent may be ignoring this component."
                )

            # --- Unexpected source: present in real but not in expected ---
            unexpected = source not in expected_percentages and real_pct > 0

            report[source] = {
                "real": real_pct,
                "expected": expected_pct,
                "difference": difference,
                "abs_difference": abs_diff,
                "status": status,
                "severity": severity,
                "recommendation": recommendation,
                "unexpected": unexpected,
            }

        return report

    def recommend_weights(
        self,
        real_percentages: Dict[str, float],
        expected_percentages: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Recommend reward weight multipliers to rebalance components.

        Key behaviour:
        - Component completely absent (real=0, expected>0): returns max weight (5.0).
          This is the reward hacking signal — the component needs aggressive upweighting.
        - Component over-represented: returns weight < 1.0 to downscale it.
        - Unexpected component (in real but not in expected): neutral weight (1.0).
        """
        if not real_percentages or not expected_percentages:
            return {}

        weights: Dict[str, float] = {}
        all_sources = set(real_percentages) | set(expected_percentages)

        for source in all_sources:
            real = real_percentages.get(source, 0.0)
            expected = expected_percentages.get(source, 0.0)

            if expected == 0:
                # Unexpected source — neutral, don't touch it
                weights[source] = 1.0
            elif real == 0:
                # Component completely absent — maximum upweight signal.
                # Returning 1.0 here would silently miss reward hacking.
                weights[source] = 5.0
            else:
                correction = expected / real
                # Apply 30% of the full correction per call to avoid
                # overcorrection while the distribution is still settling.
                weight = 1.0 + (correction - 1.0) * 0.3
                weights[source] = round(max(0.1, min(weight, 5.0)), 2)

        return weights


# ---------------------------------------------------------------------------
# Live in-loop Monitor (primary API)
# ---------------------------------------------------------------------------

class Monitor:
    """
    Live, in-loop reward monitor. Zero external dependencies.

    Drop into any training loop to detect reward component imbalances
    without needing to format or export logs.

    Example::

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

    Args:
        expected: Target distribution of reward components. Values are
            relative weights, normalized to percentages automatically.
            E.g., ``{"food": 3, "survival": 1}`` → food 75%, survival 25%.
        tolerance: Percentage-point tolerance before a component is flagged
            as imbalanced (default ±5 pp).
        window: Number of recent steps used for rolling analysis (default 200).
            Older steps are kept in memory up to ``max_history``.
        max_history: Hard cap on stored steps (default 100 000). Implemented
            as a deque so trimming is O(1) rather than O(n).
    """

    def __init__(
        self,
        expected: Dict[str, float],
        tolerance: float = 5.0,
        window: int = 200,
        max_history: int = 100_000,
    ) -> None:
        if not expected:
            raise ValueError("expected distribution must not be empty")

        for k, v in expected.items():
            _validate_finite(v, f"expected['{k}']")
            if v < 0:
                raise ValueError(
                    f"expected['{k}'] = {v} is negative. "
                    "All component weights must be >= 0."
                )

        total = sum(expected.values())
        if total <= 0:
            raise ValueError("expected values must sum to a positive number")

        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        if max_history <= 0:
            raise ValueError(f"max_history must be > 0, got {max_history}")

        self._expected: Dict[str, float] = {
            k: v / total * 100 for k, v in expected.items()
        }
        self.tolerance = tolerance
        self._window = window
        self._max_history = max_history
        # deque with maxlen gives O(1) append + automatic oldest-entry eviction
        self._history: deque = deque(maxlen=max_history)
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(
        self,
        rewards: Dict[str, float],
        episode_done: bool = False,
    ) -> None:
        """
        Record reward components for one environment step.

        Args:
            rewards: Component values for this step, e.g.
                ``{"food": 1.0, "survival": 0.01}``.
                All values must be finite numbers — NaN or Inf raises
                ``ValueError`` immediately (these indicate a broken reward
                function and must not silently corrupt analysis).
            episode_done: Unused in the free tier; accepted so that
                code written against the premium ``AutoMonitor`` API
                works unchanged when downgraded.
        """
        for comp, val in rewards.items():
            _validate_finite(val, f"rewards['{comp}']")

        self._history.append(dict(rewards))
        self._step_count += 1

    def check(self) -> AnalysisResult:
        """
        Compute a balance analysis over the current rolling window.

        Returns:
            :class:`AnalysisResult` with real vs expected percentages,
            per-component imbalance details, and suggested weight multipliers.

        Raises:
            ValueError: If no steps have been recorded yet.
        """
        if not self._history:
            raise ValueError("No steps recorded — call monitor.step() first.")

        real_pcts = self._compute_percentages()
        analyzer = RewardAnalyzer(self.tolerance)
        imbalance = analyzer.detect_imbalance(real_pcts, self._expected)
        weights = analyzer.recommend_weights(real_pcts, self._expected)

        # Determine overall severity and unexpected sources
        severity = "ok"
        unexpected: List[str] = []
        for source, info in imbalance.items():
            if info.get("unexpected"):
                unexpected.append(source)
            s = info.get("severity", "ok")
            if s == "critical":
                severity = "critical"
            elif s == "warning" and severity == "ok":
                severity = "warning"

        return AnalysisResult(
            real_percentages=real_pcts,
            expected_percentages=self._expected,
            imbalance_report=imbalance,
            suggested_reward_weights=weights,
            episode_count=self._step_count,
            sources_found=sorted(real_pcts.keys()),
            severity=severity,
            unexpected_sources=unexpected,
        )

    def print_report(self) -> None:
        """Print a formatted balance report to stdout."""
        RewardGuard(self.tolerance).print_analysis_report(self.check())

    def reset(self) -> None:
        """Clear all accumulated history without changing configuration."""
        self._history.clear()
        self._step_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Total number of steps recorded since creation or last reset."""
        return self._step_count

    @property
    def expected(self) -> Dict[str, float]:
        """Target percentage distribution (normalized, sums to 100)."""
        return dict(self._expected)

    # ------------------------------------------------------------------
    # Internal helpers (accessible to premium subclass)
    # ------------------------------------------------------------------

    def _compute_percentages(self) -> Dict[str, float]:
        """
        Compute each component's share of total reward magnitude over
        the rolling window.

        Uses ``itertools.islice(reversed(deque), window)`` for O(window)
        time rather than O(max_history) — important when max_history is large.

        Magnitude (absolute value) is used so that penalties and bonuses
        are measured by how much they dominate the signal, independent of
        sign. This matches what ``expected`` should express: the intended
        *size share* of each component.
        """
        # O(window) traversal from the end of the deque — no full copy
        recent = list(itertools.islice(reversed(self._history), self._window))

        totals: Dict[str, float] = {}
        for step_rewards in recent:
            for comp, val in step_rewards.items():
                totals[comp] = totals.get(comp, 0.0) + abs(val)

        grand_total = sum(totals.values())
        if grand_total == 0:
            return {k: 0.0 for k in totals}

        return {
            k: round(v / grand_total * 100, 2) for k, v in totals.items()
        }


# ---------------------------------------------------------------------------
# RewardGuard facade (log-based workflow)
# ---------------------------------------------------------------------------

class RewardGuard:
    """
    Log-based reward balance analysis facade.

    For live in-loop monitoring prefer :class:`Monitor`.
    """

    def __init__(self, tolerance: float = 5.0) -> None:
        self.parser = LogParser()
        self.analyzer = RewardAnalyzer(tolerance)
        self.episodes_data: List[EpisodeData] = []

    def parse_logs(self, raw_text: str) -> List[EpisodeData]:
        """Parse log text and store episode data."""
        self.episodes_data = self.parser.parse_logs(raw_text)
        return self.episodes_data

    def analyze_balance(
        self,
        parsed_data: List[EpisodeData],
        expected_percentages: Dict[str, float],
    ) -> AnalysisResult:
        """
        Analyze balance between observed and expected distributions.

        Args:
            parsed_data: List of :class:`EpisodeData` from :meth:`parse_logs`.
            expected_percentages: Target percentage per source (need not sum to
                100 — will be normalized).

        Returns:
            :class:`AnalysisResult`.
        """
        if not parsed_data:
            raise ValueError("No episode data to analyze.")

        real_percentages = self.analyzer.aggregate_percentages(parsed_data)
        imbalance_report = self.analyzer.detect_imbalance(
            real_percentages, expected_percentages
        )
        suggested_weights = self.analyzer.recommend_weights(
            real_percentages, expected_percentages
        )

        severity = "ok"
        unexpected: List[str] = []
        for source, info in imbalance_report.items():
            if info.get("unexpected"):
                unexpected.append(source)
            s = info.get("severity", "ok")
            if s == "critical":
                severity = "critical"
            elif s == "warning" and severity == "ok":
                severity = "warning"

        return AnalysisResult(
            real_percentages=real_percentages,
            expected_percentages=expected_percentages,
            imbalance_report=imbalance_report,
            suggested_reward_weights=suggested_weights,
            episode_count=len(parsed_data),
            sources_found=sorted(real_percentages.keys()),
            severity=severity,
            unexpected_sources=unexpected,
        )

    def recommend_weights(
        self,
        real_percentages: Dict[str, float],
        expected_percentages: Dict[str, float],
    ) -> Dict[str, float]:
        """Convenience wrapper for :meth:`RewardAnalyzer.recommend_weights`."""
        return self.analyzer.recommend_weights(real_percentages, expected_percentages)

    def print_analysis_report(self, result: AnalysisResult) -> None:
        """Print a formatted analysis report to stdout."""
        print("\n" + "=" * 60)
        print("REWARDGUARD ANALYSIS REPORT")
        if result.severity != "ok":
            print(f"  *** OVERALL SEVERITY: {result.severity.upper()} ***")
        print("=" * 60)

        print(f"\n  Episodes analyzed : {result.episode_count}")
        print(f"  Sources found     : {', '.join(result.sources_found)}")

        if result.unexpected_sources:
            print(f"  Unexpected sources: {', '.join(result.unexpected_sources)}  [not in expected]")

        print(f"\n  {'Source':<15} {'Real %':<10} {'Expected %':<12} {'Diff':<10} Severity")
        print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*10} --------")

        for source in result.sources_found:
            real = result.real_percentages.get(source, 0.0)
            expected = result.expected_percentages.get(source, 0.0)
            diff = result.imbalance_report[source]["difference"]
            severity = result.imbalance_report[source]["severity"]
            symbol = {"ok": "OK", "warning": "WARNING", "critical": "CRITICAL"}.get(severity, severity)
            print(
                f"  {source:<15} {real:<10.1f} {expected:<12.1f} "
                f"{diff:>+9.1f}  {symbol}"
            )

        print("\n  Suggested weight multipliers:")
        for source, weight in result.suggested_reward_weights.items():
            flag = ""
            sev = result.imbalance_report.get(source, {}).get("severity", "ok")
            if sev == "critical":
                flag = "  <-- CRITICAL"
            elif sev == "warning":
                flag = "  <-- ADJUST"
            print(f"    {source}: {weight:.2f}x{flag}")

        print("\n  Actions needed:")
        any_action = False
        for source, info in result.imbalance_report.items():
            if info["status"] == "imbalanced":
                print(f"    • {source}: {info['recommendation']}")
                any_action = True
        if not any_action:
            print("    All reward sources are well balanced.")

        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def parse_logs(raw_text: str) -> List[EpisodeData]:
    """Parse training logs from raw text."""
    return LogParser().parse_logs(raw_text)


def analyze_balance(
    parsed_data: List[EpisodeData],
    expected_percentages: Dict[str, float],
) -> AnalysisResult:
    """Analyze balance between observed and expected distributions."""
    return RewardGuard().analyze_balance(parsed_data, expected_percentages)


def recommend_weights(
    real_percentages: Dict[str, float],
    expected_percentages: Dict[str, float],
) -> Dict[str, float]:
    """Recommend reward weight multipliers."""
    return RewardAnalyzer().recommend_weights(real_percentages, expected_percentages)
