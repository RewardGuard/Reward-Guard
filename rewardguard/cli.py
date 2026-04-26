#!/usr/bin/env python3
"""
RewardGuard CLI — analyze reward balance from training log files.

Usage examples:

    # Analyze a log file with known expected distribution
    rewardguard analyze training.log --expected food:60 survival:40

    # Analyze and show only imbalanced components
    rewardguard analyze training.log --expected food:60 survival:40 --imbalanced-only

    # Read from stdin
    cat training.log | rewardguard analyze - --expected task:70 safety:30
"""

import argparse
import csv
import json
import sys
from typing import Dict

from .Core import LogParser, RewardGuard, RewardAnalyzer


def _parse_expected(pairs: list) -> Dict[str, float]:
    """
    Parse 'key:value' strings into a percentage dict.

    Raises:
        SystemExit: On invalid format.
    """
    result: Dict[str, float] = {}
    for pair in pairs:
        try:
            k, v = pair.split(":", 1)
            result[k.strip()] = float(v.strip())
        except (ValueError, AttributeError):
            print(
                f"error: invalid --expected entry '{pair}' "
                "(expected format: name:value, e.g. food:60)",
                file=sys.stderr,
            )
            sys.exit(1)
    return result


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run the analyze sub-command."""
    # Read input
    if args.logfile == "-":
        raw_text = sys.stdin.read()
    else:
        try:
            with open(args.logfile, encoding="utf-8") as fh:
                raw_text = fh.read()
        except FileNotFoundError:
            print(f"error: file not found: {args.logfile}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"error reading file: {exc}", file=sys.stderr)
            return 1

    # Parse logs
    parser = LogParser()
    episodes = parser.parse_logs(raw_text)
    if not episodes:
        print(
            "error: no episode data found in log.\n"
            "Make sure your logs follow the expected format:\n"
            "  Ep 1 | SUCCESS | reward=15.3\n"
            "  === EPISODE DATA ===\n"
            "  sources:\n"
            "    food: 12.1\n"
            "    survival: 3.2",
            file=sys.stderr,
        )
        return 1

    # Build expected distribution
    if args.expected:
        expected = _parse_expected(args.expected)
    else:
        # Auto-detect: treat current distribution as expected (just show stats)
        analyzer = RewardAnalyzer()
        real = analyzer.aggregate_percentages(episodes)
        expected = real
        print(
            "[info] No --expected provided — showing observed distribution only.\n"
            "       Rerun with --expected source:pct ... to detect imbalances.\n"
        )

    # Analyze
    guard = RewardGuard(tolerance=args.tolerance)
    result = guard.analyze_balance(episodes, expected)

    output_format = getattr(args, "output", "text")

    if output_format == "json":
        out = {
            "episode_count": result.episode_count,
            "severity": result.severity,
            "sources_found": result.sources_found,
            "unexpected_sources": result.unexpected_sources,
            "real_percentages": result.real_percentages,
            "expected_percentages": result.expected_percentages,
            "suggested_reward_weights": result.suggested_reward_weights,
            "imbalance_report": result.imbalance_report,
        }
        print(json.dumps(out, indent=2))
        return 0

    if output_format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["source", "real_pct", "expected_pct", "difference", "severity", "weight", "recommendation"])
        for source in result.sources_found:
            info = result.imbalance_report[source]
            writer.writerow([
                source,
                f"{info['real']:.2f}",
                f"{info['expected']:.2f}",
                f"{info['difference']:+.2f}",
                info["severity"],
                f"{result.suggested_reward_weights.get(source, 1.0):.2f}",
                info["recommendation"],
            ])
        return 0

    # Default: text output
    if args.imbalanced_only:
        imbalanced = {
            s: v
            for s, v in result.imbalance_report.items()
            if v["status"] == "imbalanced"
        }
        if not imbalanced:
            print("All reward sources are balanced within tolerance.")
            return 0
        print("\nIMBALANCED COMPONENTS:")
        for source, info in imbalanced.items():
            print(
                f"  {source}: real={info['real']:.1f}%  "
                f"expected={info['expected']:.1f}%  "
                f"diff={info['difference']:+.1f}pp  "
                f"severity={info['severity']}  "
                f"→ {info['recommendation']}"
            )
        print()
    else:
        guard.print_analysis_report(result)

    return 0


def main() -> None:
    """Entry point for the `rewardguard` CLI command."""
    parser = argparse.ArgumentParser(
        prog="rewardguard",
        description="Detect reward component imbalances in RL training logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # --- analyze ---
    analyze_parser = sub.add_parser(
        "analyze",
        help="Analyze a training log file for reward imbalances.",
    )
    analyze_parser.add_argument(
        "logfile",
        help="Path to the training log file, or '-' to read from stdin.",
    )
    analyze_parser.add_argument(
        "--expected",
        nargs="+",
        metavar="name:pct",
        help=(
            "Expected percentage per reward source. "
            "E.g.: --expected food:60 survival:40"
        ),
    )
    analyze_parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        metavar="PP",
        help="Percentage-point tolerance before flagging imbalance (default: 5.0).",
    )
    analyze_parser.add_argument(
        "--imbalanced-only",
        action="store_true",
        help="Print only components that are outside tolerance.",
    )
    analyze_parser.add_argument(
        "--output",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text).",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
