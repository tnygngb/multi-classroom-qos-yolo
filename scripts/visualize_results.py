"""Generate publication-ready charts from benchmark outputs."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.config import ensure_output_dirs, load_config, resolve_path
from src.common.logger import configure_logger_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize benchmark results into chart images.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base config YAML.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSON path (multi benchmark or export summary).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reports",
        help="Directory for generated charts.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="Stage-7 Benchmark",
        help="Prefix used in figure titles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate input/output paths.",
    )
    return parser


def _latest_file_by_pattern(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Input JSON root must be object: {path}")
    return payload


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_multi_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("multi_table"), list):
        rows = payload.get("multi_table", [])
        return [dict(item) for item in rows if isinstance(item, Mapping)]

    records = payload.get("records", [])
    if isinstance(records, list):
        rows: list[dict[str, Any]] = []
        for item in records:
            if not isinstance(item, Mapping):
                continue
            metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), Mapping) else {}
            latency = metrics.get("latency_ms", {}) if isinstance(metrics.get("latency_ms"), Mapping) else {}
            fairness = metrics.get("fairness", {}) if isinstance(metrics.get("fairness"), Mapping) else {}
            rows.append(
                {
                    "strategy": item.get("strategy"),
                    "stream_count": item.get("stream_count"),
                    "status": item.get("status"),
                    "total_fps": metrics.get("total_fps"),
                    "p95_latency_ms": latency.get("p95_ms"),
                    "effective_detection_score": metrics.get("effective_detection_score"),
                    "fairness_jain_index": fairness.get("jain_index"),
                }
            )
        return rows

    nested_multi = payload.get("multi_stream")
    if isinstance(nested_multi, Mapping):
        return _extract_multi_rows(nested_multi)

    return []


def _extract_single_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("single_table"), list):
        rows = payload.get("single_table", [])
        return [dict(item) for item in rows if isinstance(item, Mapping)]

    if isinstance(payload.get("records"), list):
        records = payload.get("records", [])
        variants = [dict(item) for item in records if isinstance(item, Mapping) and "variant" in item]
        if variants:
            return variants

    nested_single = payload.get("single_stream")
    if isinstance(nested_single, Mapping):
        return _extract_single_rows(nested_single)

    if isinstance(payload.get("results"), list):
        return [dict(item) for item in payload.get("results", []) if isinstance(item, Mapping)]

    return []


def _group_metric(rows: list[dict[str, Any]], key: str) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        status = str(row.get("status", "completed"))
        if status not in {"completed", "", "None"}:
            continue

        strategy = str(row.get("strategy", "unknown"))
        stream_count = _to_int(row.get("stream_count"))
        value = _to_float(row.get(key))
        if stream_count is None or value is None:
            continue

        grouped.setdefault(strategy, []).append((stream_count, value))

    for strategy, points in grouped.items():
        grouped[strategy] = sorted(points, key=lambda item: item[0])
    return grouped


def _plot_line_chart(
    *,
    grouped: dict[str, list[tuple[int, float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)

    marker_cycle = ["o", "s", "^", "D", "v"]
    for index, (strategy, points) in enumerate(sorted(grouped.items())):
        if not points:
            continue
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(xs, ys, marker=marker_cycle[index % len(marker_cycle)], linewidth=2.0, label=strategy)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_single_variant_bar(*, rows: list[dict[str, Any]], title_prefix: str, output_path: Path) -> None:
    if not rows:
        return

    variants: list[str] = []
    fps_values: list[float] = []
    p95_values: list[float] = []

    for row in rows:
        variant = str(row.get("variant", "unknown"))
        fps = _to_float(row.get("fps"))
        p95 = _to_float(row.get("p95_latency_ms"))
        if fps is None or p95 is None:
            continue
        variants.append(variant)
        fps_values.append(fps)
        p95_values.append(p95)

    if not variants:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), dpi=160)

    axes[0].bar(variants, fps_values, color="#2a9d8f")
    axes[0].set_title("Single-Stream FPS")
    axes[0].set_ylabel("FPS")

    axes[1].bar(variants, p95_values, color="#e76f51")
    axes[1].set_title("Single-Stream p95 Latency")
    axes[1].set_ylabel("Latency (ms)")

    fig.suptitle(f"{title_prefix} | Single-Stream Comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    config = ensure_output_dirs(load_config(args.config))
    logger = configure_logger_from_config(config, logger_name="visualize_results")

    output_dir = Path(resolve_path(args.output_dir, base_dir=config["_meta"]["project_root"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(resolve_path(args.input, base_dir=config["_meta"]["project_root"]))
    else:
        reports_dir = Path(str(config["paths"]["reports"]))
        input_path = _latest_file_by_pattern(reports_dir, "experiment_summary_*.json")
        if input_path is None:
            input_path = _latest_file_by_pattern(reports_dir, "multi_stream_benchmark_*.json")

    if input_path is None or not input_path.exists():
        raise FileNotFoundError("No benchmark JSON found for visualization.")

    logger.info("Visualization input: %s", input_path)
    logger.info("Visualization output_dir: %s", output_dir)

    if args.dry_run:
        logger.info("Dry-run finished. No charts were generated.")
        return 0

    payload = _load_json(input_path)
    multi_rows = _extract_multi_rows(payload)
    single_rows = _extract_single_rows(payload)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    throughput_png = output_dir / f"stage7_streams_vs_throughput_{timestamp}.png"
    latency_png = output_dir / f"stage7_streams_vs_latency_{timestamp}.png"
    effective_png = output_dir / f"stage7_streams_vs_effective_detection_{timestamp}.png"
    single_png = output_dir / f"stage7_single_stream_compare_{timestamp}.png"

    throughput_grouped = _group_metric(multi_rows, "total_fps")
    latency_grouped = _group_metric(multi_rows, "p95_latency_ms")
    effective_grouped = _group_metric(multi_rows, "effective_detection_score")

    _plot_line_chart(
        grouped=throughput_grouped,
        title=f"{args.title_prefix} | Stream Count vs Throughput",
        xlabel="Concurrent Streams",
        ylabel="Total FPS",
        output_path=throughput_png,
    )
    _plot_line_chart(
        grouped=latency_grouped,
        title=f"{args.title_prefix} | Stream Count vs p95 Latency",
        xlabel="Concurrent Streams",
        ylabel="Latency (ms)",
        output_path=latency_png,
    )
    _plot_line_chart(
        grouped=effective_grouped,
        title=f"{args.title_prefix} | Stream Count vs Effective Detection Score",
        xlabel="Concurrent Streams",
        ylabel="Effective Detection Score",
        output_path=effective_png,
    )

    _plot_single_variant_bar(rows=single_rows, title_prefix=args.title_prefix, output_path=single_png)

    logger.info("Generated charts:")
    logger.info("  %s", throughput_png)
    logger.info("  %s", latency_png)
    logger.info("  %s", effective_png)
    if single_rows:
        logger.info("  %s", single_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



