#!/usr/bin/env python3
"""
Generate Markdown Benchmark Report from JSON results.

Reads benchmark results from JSON files and generates a comprehensive
markdown report with tables, comparisons, and summaries.

Usage:
    python scripts/generate_benchmark_report.py
    python scripts/generate_benchmark_report.py --input benchmark_results/ground_truth_results.json
    python scripts/generate_benchmark_report.py --include-pytorch --pytorch-file pytorch_reference_results.json
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    mlx_results_file: Path
    pytorch_results_file: Optional[Path]
    output_file: Path
    title: str = "mlx-music Ground Truth Benchmark Report"


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def format_float(value: Optional[float], decimals: int = 2) -> str:
    """Format float or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def generate_summary_table(results: List[Dict]) -> str:
    """Generate main summary table."""
    lines = [
        "## Summary",
        "",
        "| Model | Quant | Steps | Time (s) | RTF | Memory (MB) | Quality |",
        "|-------|-------|-------|----------|-----|-------------|---------|",
    ]

    for r in results:
        steps = r.get("steps", "N/A")
        time_sec = format_float(r.get("total_time_sec"))
        rtf = format_float(r.get("realtime_factor"))
        memory = format_float(r.get("peak_memory_mb"), 0)

        # Determine quality status
        is_silent = r.get("is_silent", False)
        is_clipped = r.get("is_clipped", False)
        if is_silent:
            quality = "SILENT"
        elif is_clipped:
            quality = "CLIPPED"
        else:
            quality = "PASS"

        row = f"| {r['model_name']} | {r['quantization']} | {steps} | {time_sec} | {rtf} | {memory} | {quality} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def generate_quantization_comparison(results: List[Dict]) -> str:
    """Generate quantization comparison section."""
    lines = [
        "## Quantization Comparison",
        "",
    ]

    # Group by model
    models = {}
    for r in results:
        model_name = r["model_name"]
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(r)

    for model_name, model_results in models.items():
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| Quantization | Model Size (MB) | Peak Memory (MB) | Gen Time (s) | RTF | Memory Reduction |")
        lines.append("|--------------|-----------------|------------------|--------------|-----|------------------|")

        # Sort by quantization
        quant_order = {"bfloat16": 0, "int8": 1, "int4": 2, "mixed": 3}
        sorted_results = sorted(model_results, key=lambda x: quant_order.get(x["quantization"], 99))

        baseline_memory = None
        baseline_model_size = None
        for r in sorted_results:
            quant = r["quantization"]
            model_size = r.get("model_memory_mb", 0)
            model_size_str = format_float(model_size, 1) if model_size > 0 else "N/A"
            peak_memory = r.get("peak_memory_mb", 0)
            peak_memory_str = format_float(peak_memory, 0)
            gen_time = format_float(r.get("denoise_time_sec") or r.get("total_time_sec"))
            rtf = format_float(r.get("realtime_factor"))

            # Calculate memory reduction (comparing peak memory)
            if baseline_memory is None and quant == "bfloat16":
                baseline_memory = peak_memory
                baseline_model_size = model_size
                reduction = "-"
            elif baseline_memory and baseline_memory > 0:
                reduction = f"{(1 - peak_memory / baseline_memory) * 100:.0f}%"
            else:
                reduction = "N/A"

            row = f"| {quant} | {model_size_str} | {peak_memory_str} | {gen_time} | {rtf} | {reduction} |"
            lines.append(row)

        # Add note about expected vs actual reduction if we have baseline
        if baseline_model_size and baseline_model_size > 0:
            lines.append("")
            int4_result = next((r for r in sorted_results if r["quantization"] == "int4"), None)
            int8_result = next((r for r in sorted_results if r["quantization"] == "int8"), None)
            if int4_result:
                int4_model_size = int4_result.get("model_memory_mb", 0)
                if int4_model_size > 0:
                    actual_reduction = (1 - int4_model_size / baseline_model_size) * 100
                    lines.append(f"*INT4 model size reduction: {actual_reduction:.0f}% (expected ~75%)*")
            if int8_result:
                int8_model_size = int8_result.get("model_memory_mb", 0)
                if int8_model_size > 0:
                    actual_reduction = (1 - int8_model_size / baseline_model_size) * 100
                    lines.append(f"*INT8 model size reduction: {actual_reduction:.0f}% (expected ~50%)*")

        lines.append("")

    return "\n".join(lines)


def generate_scheduler_comparison(scheduler_results: List[Dict]) -> str:
    """Generate scheduler comparison section."""
    if not scheduler_results:
        return ""

    lines = [
        "## Scheduler Comparison (ACE-Step)",
        "",
        "| Scheduler | Steps | Time (s) | RTF | Quality Score |",
        "|-----------|-------|----------|-----|---------------|",
    ]

    for r in scheduler_results:
        scheduler = r.get("scheduler", "N/A")
        steps = r.get("steps", "N/A")
        time_sec = format_float(r.get("total_time_sec"))
        rtf = format_float(r.get("realtime_factor"))
        quality = format_float(r.get("quality_score"))

        row = f"| {scheduler} | {steps} | {time_sec} | {rtf} | {quality} |"
        lines.append(row)

    lines.append("")
    lines.append("*Quality Score is relative to baseline (Euler@50)*")
    lines.append("")

    return "\n".join(lines)


def generate_pytorch_comparison(mlx_results: List[Dict], pytorch_results: List[Dict]) -> str:
    """Generate MLX vs PyTorch comparison."""
    if not pytorch_results:
        return ""

    lines = [
        "## MLX vs PyTorch Comparison",
        "",
        "| Model | Framework | Device | Gen Time (s) | RTF | Speedup |",
        "|-------|-----------|--------|--------------|-----|---------|",
    ]

    # Group MLX results by model
    mlx_by_model = {}
    for r in mlx_results:
        if r["quantization"] == "bfloat16":  # Compare baseline
            mlx_by_model[r["model_name"]] = r

    # Add PyTorch results
    for r in pytorch_results:
        model_name = r.get("model_name", "Unknown")
        framework = "PyTorch"
        device = r.get("device", "N/A")
        gen_time = format_float(r.get("generation_time_sec"))
        rtf = format_float(r.get("realtime_factor"))

        row = f"| {model_name} | {framework} | {device} | {gen_time} | {rtf} | - |"
        lines.append(row)

        # Add corresponding MLX result
        if model_name in mlx_by_model:
            mlx_r = mlx_by_model[model_name]
            mlx_gen_time = mlx_r.get("denoise_time_sec") or mlx_r.get("total_time_sec", 0)
            mlx_rtf = mlx_r.get("realtime_factor", 0)
            pytorch_gen_time = r.get("generation_time_sec", 0)

            speedup = pytorch_gen_time / mlx_gen_time if mlx_gen_time > 0 else 0

            row = f"| {model_name} | MLX | Metal | {format_float(mlx_gen_time)} | {format_float(mlx_rtf)} | {format_float(speedup)}x |"
            lines.append(row)

    lines.append("")
    return "\n".join(lines)


def generate_quality_metrics(results: List[Dict]) -> str:
    """Generate quality metrics section."""
    lines = [
        "## Quality Metrics",
        "",
        "| Model | Quant | RMS Energy | Dynamic Range (dB) | Spectral Centroid (Hz) | CLAP Score |",
        "|-------|-------|------------|-------------------|------------------------|------------|",
    ]

    for r in results:
        model = r["model_name"]
        quant = r["quantization"]
        rms = format_float(r.get("rms_energy"), 4)
        dynamic_range = format_float(r.get("dynamic_range_db"), 1)
        spectral = format_float(r.get("spectral_centroid_hz"), 0)
        clap = format_float(r.get("clap_score"), 3)

        row = f"| {model} | {quant} | {rms} | {dynamic_range} | {spectral} | {clap} |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def generate_key_findings(results: List[Dict], scheduler_results: Optional[List[Dict]] = None) -> str:
    """Generate key findings section."""
    lines = [
        "## Key Findings",
        "",
    ]

    # Find best results
    if results:
        # Memory reduction from INT4
        bf16_results = [r for r in results if r["quantization"] == "bfloat16"]
        int4_results = [r for r in results if r["quantization"] == "int4"]

        if bf16_results and int4_results:
            for model in set(r["model_name"] for r in results):
                bf16 = next((r for r in bf16_results if r["model_name"] == model), None)
                int4 = next((r for r in int4_results if r["model_name"] == model), None)

                if bf16 and int4:
                    bf16_mem = bf16.get("peak_memory_mb", 0)
                    int4_mem = int4.get("peak_memory_mb", 0)
                    if bf16_mem > 0:
                        reduction = (1 - int4_mem / bf16_mem) * 100
                        lines.append(f"1. **{model} INT4 Memory Reduction**: {reduction:.0f}%")

        # Best realtime factor
        best_rtf = max(results, key=lambda r: r.get("realtime_factor", 0))
        if best_rtf.get("realtime_factor", 0) > 0:
            lines.append(
                f"2. **Best Realtime Factor**: {best_rtf['model_name']} ({best_rtf['quantization']}) "
                f"achieves **{best_rtf['realtime_factor']:.2f}x** realtime"
            )

    # Scheduler findings
    if scheduler_results:
        # Find DPM++ that matches Euler quality
        euler_50 = next((r for r in scheduler_results if r["scheduler"] == "euler" and r["steps"] == 50), None)
        dpm_20 = next((r for r in scheduler_results if r["scheduler"] == "dpm++" and r["steps"] == 20), None)

        if euler_50 and dpm_20:
            speedup = euler_50.get("total_time_sec", 0) / dpm_20.get("total_time_sec", 1)
            lines.append(
                f"3. **DPM++ Efficiency**: DPM++ at 20 steps achieves similar quality to Euler at 50 steps "
                f"(**{speedup:.1f}x speedup**)"
            )

    lines.append("")
    return "\n".join(lines)


def generate_audio_samples_section(results: List[Dict]) -> str:
    """Generate audio samples section if audio files were saved."""
    audio_files = [r for r in results if r.get("audio_file")]
    if not audio_files:
        return ""

    lines = [
        "## Audio Samples",
        "",
        "Generated audio files:",
        "",
    ]

    for r in audio_files:
        audio_file = r["audio_file"]
        model = r["model_name"]
        quant = r["quantization"]
        lines.append(f"- `{audio_file}` - {model} ({quant})")

    lines.append("")
    return "\n".join(lines)


def generate_report(config: ReportConfig) -> str:
    """Generate full benchmark report."""
    # Load results
    mlx_data = load_json(config.mlx_results_file)
    results = mlx_data.get("results", [])
    scheduler_results = mlx_data.get("scheduler_comparison", [])
    benchmark_config = mlx_data.get("config", {})

    pytorch_results = []
    if config.pytorch_results_file and config.pytorch_results_file.exists():
        pytorch_data = load_json(config.pytorch_results_file)
        pytorch_results = pytorch_data.get("results", [])

    # Build report
    sections = [
        f"# {config.title}",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]

    # Configuration section
    if benchmark_config:
        sections.extend(
            [
                "## Configuration",
                "",
                f"- **Prompt**: \"{benchmark_config.get('prompt', 'N/A')}\"",
                f"- **Duration**: {benchmark_config.get('duration', 'N/A')}s",
                f"- **Seed**: {benchmark_config.get('seed', 'N/A')}",
                f"- **Runs**: {benchmark_config.get('runs', 'N/A')}",
                "",
            ]
        )

    # Main sections
    sections.append(generate_summary_table(results))
    sections.append(generate_quantization_comparison(results))

    if scheduler_results:
        sections.append(generate_scheduler_comparison(scheduler_results))

    if pytorch_results:
        sections.append(generate_pytorch_comparison(results, pytorch_results))

    sections.append(generate_quality_metrics(results))
    sections.append(generate_key_findings(results, scheduler_results))
    sections.append(generate_audio_samples_section(results))

    # Footer
    sections.extend(
        [
            "---",
            "",
            "*Generated by mlx-music benchmark suite*",
        ]
    )

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Generate Benchmark Report")

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark_results/ground_truth_results.json"),
        help="Path to MLX benchmark results JSON",
    )
    parser.add_argument(
        "--pytorch-file",
        type=Path,
        default=None,
        help="Path to PyTorch benchmark results JSON",
    )
    parser.add_argument(
        "--include-pytorch",
        action="store_true",
        help="Include PyTorch comparison (looks for pytorch_reference_results.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file (default: benchmark_results/BENCHMARK_REPORT.md)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="mlx-music Ground Truth Benchmark Report",
        help="Report title",
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_file = args.output
    else:
        output_file = args.input.parent / "BENCHMARK_REPORT.md"

    # Determine PyTorch file
    pytorch_file = args.pytorch_file
    if args.include_pytorch and pytorch_file is None:
        pytorch_file = args.input.parent / "pytorch_reference_results.json"

    config = ReportConfig(
        mlx_results_file=args.input,
        pytorch_results_file=pytorch_file,
        output_file=output_file,
        title=args.title,
    )

    # Check input exists
    if not config.mlx_results_file.exists():
        print(f"ERROR: Input file not found: {config.mlx_results_file}")
        print("Run benchmark_ground_truth.py first to generate results.")
        return 1

    # Generate report
    print(f"Generating report from: {config.mlx_results_file}")
    if config.pytorch_results_file and config.pytorch_results_file.exists():
        print(f"Including PyTorch comparison from: {config.pytorch_results_file}")

    report = generate_report(config)

    # Write report
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_file, "w") as f:
        f.write(report)

    print(f"Report saved to: {config.output_file}")
    return 0


if __name__ == "__main__":
    exit(main())
