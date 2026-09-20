"""
Read moe_stats.csv and compute routing balance metrics + plots.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# Load CSV
# ==============================================================================
def load_moe_stats(csv_path: str | Path) -> list[dict]:
    """Load moe_stats.csv written by LLMTrainer."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "step": int(row["step"]),
                "layer": int(row["layer"]),
                "expert": int(row["expert"]),
                "token_distr": float(row["token_distr"]),
                "token_dropped": float(row["token_dropped"]),
            })
    return rows


def _token_distr_by_step_layer(rows: list[dict], n_experts: int) -> dict:
    """
    Build lookup: step -> layer -> np.array of shape (n_experts,)
    """
    out = {}
    for row in rows:
        step = row["step"]
        layer = row["layer"]
        expert = row["expert"]
        out.setdefault(step, {})
        out[step].setdefault(layer, np.zeros(n_experts, dtype=np.float64))
        out[step][layer][expert] = row["token_distr"]
    return out


def _drop_pct_by_step(rows: list[dict], n_layers: int, batch_size: int, ctx_len: int, topk: int) -> dict:
    """
    Drop rate (%) per step, averaged over MoE layers.
    Same idea as trainer._log_moe_stats().
    """
    routed_per_layer = batch_size * ctx_len * topk
    step_layer_dropped = {}
    for row in rows:
        key = (row["step"], row["layer"])
        step_layer_dropped[key] = step_layer_dropped.get(key, 0.0) + row["token_dropped"]

    step_drop = {}
    for (step, layer), dropped in step_layer_dropped.items():
        pct = 100.0 * dropped / routed_per_layer
        step_drop.setdefault(step, []).append(pct)

    return {step: float(np.mean(vals)) for step, vals in step_drop.items()}


# ==============================================================================
# Metrics
# ==============================================================================
def fair_share(n_experts: int) -> float:
    """Uniform target per expert. With 4 experts -> 0.25."""
    return 1.0 / n_experts


def layer_cv(token_distr: np.ndarray) -> float:
    """Coefficient of variation across experts in one layer."""
    return float(np.std(token_distr) / (np.mean(token_distr) + 1e-5))


def layer_max_load_ratio(token_distr: np.ndarray, n_experts: int) -> float:
    """
    Hottest expert vs fair share.
    1.0 = perfect. 1.8 = one expert gets 1.8x its fair share.
    """
    return float(np.max(token_distr) / fair_share(n_experts))


def aggregate_at_step(stats: dict, step: int) -> dict | None:
    """
    Mean CV and mean max_load_ratio over all layers at one step.
    Returns None if step not logged.
    """
    if step not in stats:
        return None

    n_experts = len(next(iter(stats[step].values())))
    cvs = []
    max_ratios = []
    layer_min_med_max = {}

    for layer, distr in sorted(stats[step].items()):
        cvs.append(layer_cv(distr))
        max_ratios.append(layer_max_load_ratio(distr, n_experts))
        layer_min_med_max[layer] = (
            float(np.min(distr)),
            float(np.median(distr)),
            float(np.max(distr)),
        )

    return {
        "cv": float(np.mean(cvs)),
        "max_load_ratio": float(np.mean(max_ratios)),
        "layer_min_med_max": layer_min_med_max,
        "worst_layer": max(layer_min_med_max, key=lambda l: layer_min_med_max[l][2]),
        "worst_layer_max": layer_min_med_max[max(layer_min_med_max, key=lambda l: layer_min_med_max[l][2])][2],
    }


def cv_max_over_steps(stats: dict) -> tuple[list[int], list[float], list[float]]:
    """Aggregated CV and max_load_ratio at every logged step (for line plots)."""
    steps = sorted(stats.keys())
    cvs = []
    max_ratios = []
    for step in steps:
        agg = aggregate_at_step(stats, step)
        cvs.append(agg["cv"])
        max_ratios.append(agg["max_load_ratio"])
    return steps, cvs, max_ratios


def analyze_run(
    csv_path: str | Path,
    n_experts: int,
    n_layers: int,
    batch_size: int,
    ctx_len: int,
    topk: int,
    checkpoint_steps: list[int],
) -> dict:
    """Full metrics for one run."""
    rows = load_moe_stats(csv_path)
    stats = _token_distr_by_step_layer(rows, n_experts)
    drop_by_step = _drop_pct_by_step(rows, n_layers, batch_size, ctx_len, topk)

    checkpoints = {}
    for step in checkpoint_steps:
        checkpoints[step] = aggregate_at_step(stats, step)

    steps, cv_series, max_series = cv_max_over_steps(stats)

    # Drift: late checkpoint vs early (step 50)
    early_step = checkpoint_steps[0]
    late_step = checkpoint_steps[-1]
    early = checkpoints.get(early_step)
    late = checkpoints.get(late_step)

    drift_cv = None
    drift_max = None
    if early and late:
        drift_cv = late["cv"] - early["cv"]
        drift_max = late["max_load_ratio"] - early["max_load_ratio"]

    late_drop = drop_by_step.get(late_step, 0.0)

    return {
        "steps": steps,
        "cv_series": cv_series,
        "max_load_series": max_series,
        "checkpoints": checkpoints,
        "drift_cv": drift_cv,
        "drift_max": drift_max,
        "drop_pct_late": late_drop,
    }


# ==============================================================================
# Print helpers
# ==============================================================================
def print_run_report(run_name: str, metrics: dict, checkpoint_steps: list[int], n_experts: int) -> None:
    """Console summary for one run."""
    target = fair_share(n_experts)

    print("\n" + "=" * 64)
    print(f"Run: {run_name}")
    print("=" * 64)

    print("\n--- Aggregated routing (mean over layers) ---")
    for step in checkpoint_steps:
        agg = metrics["checkpoints"].get(step)
        if agg is None:
            print(f"  step {step:4d}:  (not logged)")
            continue
        print(
            f"  step {step:4d}:  CV={agg['cv']:.3f}   "
            f"max_load_ratio={agg['max_load_ratio']:.2f}   "
            f"target={target:.3f}"
        )

    if metrics["drift_cv"] is not None:
        print(f"  drift CV:  {metrics['drift_cv']:+.3f}")
        print(f"  drift max: {metrics['drift_max']:+.3f}")

    late_step = checkpoint_steps[-1]
    late = metrics["checkpoints"].get(late_step)
    if late:
        print(f"\n--- Layer-wise @ step {late_step} (min / median / max) ---")
        print(f"  target per expert = {target:.3f}")
        for layer, (mn, med, mx) in sorted(late["layer_min_med_max"].items()):
            print(f"  L{layer}:  {mn:.2f} / {med:.2f} / {mx:.2f}")
        print(f"  worst layer: L{late['worst_layer']} (max={late['worst_layer_max']:.2f})")

    print(f"\n  drop rate @ step {late_step}: {metrics['drop_pct_late']:.1f}%")


# ==============================================================================
# Plots
# ==============================================================================
def plot_layer_token_distr_band(
    csv_path: str | Path,
    n_experts: int,
    save_path: str | Path,
    target: float | None = None,
) -> None:
    """
    Per-run plot: min / median / max token_distr across experts, one subplot per layer.
    Layout: 2 layers per row.
    """
    rows = load_moe_stats(csv_path)
    stats = _token_distr_by_step_layer(rows, n_experts)
    if target is None:
        target = fair_share(n_experts)
    layers = sorted({row["layer"] for row in rows})
    steps = sorted(stats.keys())
    n_layers = len(layers)
    # 2 subplots per row; figsize grows with layer count
    n_cols = 2
    n_rows = int(np.ceil(n_layers / n_cols))
    fig_w = 5 * n_cols          # 10 inches wide
    fig_h = 3.5 * n_rows        # ~3.5 inches per row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle("Token distribution band per layer (min / median / max)", fontsize=12, y=1.02)
    for i, layer in enumerate(layers):
        ax = axes[i // n_cols][i % n_cols]
        mins, meds, maxs = [], [], []
        for step in steps:
            distr = stats[step][layer]
            mins.append(np.min(distr))
            meds.append(np.median(distr))
            maxs.append(np.max(distr))
        ax.fill_between(steps, mins, maxs, alpha=0.25, label="min–max")
        ax.plot(steps, meds, marker=".", markersize=3, label="median")
        ax.axhline(target, color="gray", linestyle="--", linewidth=1, label="uniform")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("step")
        ax.set_ylabel("token_distr")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    # Hide unused subplots
    for j in range(n_layers, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_config_comparison(summary_rows: list[dict], save_path: str | Path) -> None:
    """Bar chart: CV and max_load_ratio @ final step for all configs."""
    names = [r["run_name"] for r in summary_rows]
    cvs = [r["cv_late"] for r in summary_rows]
    maxs = [r["max_ratio_late"] for r in summary_rows]

    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 2 * len(names)), 4))
    x = np.arange(len(names))

    axes[0].bar(x, cvs, color="steelblue")
    axes[0].set_title("Routing CV @ final step (lower = better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)

    axes[1].bar(x, maxs, color="coral")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ideal = 1.0")
    axes[1].set_title("Max load ratio @ final step (closer to 1.0 = better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_metric_over_steps(
    all_runs: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    save_path: str | Path,
    ideal_line: float | None = None,
) -> None:
    """Line plot: one curve per config (aggregated metric vs step)."""
    n_configs = len(all_runs)
    n_cols = min(4, max(1, n_configs))
    n_items = n_configs + (1 if ideal_line is not None else 0)
    n_legend_rows = int(np.ceil(n_items / n_cols))

    fig_h = 4.5 + 0.45 * n_legend_rows
    fig, ax = plt.subplots(figsize=(10, fig_h))

    for run in all_runs:
        ax.plot(
            run["steps"], run[metric_key],
            marker=".", markersize=3, label=run["run_name"],
        )

    if ideal_line is not None:
        ax.axhline(ideal_line, color="gray", linestyle="--", linewidth=1, label="ideal")

    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)

    # get_legend_handles_labels() works without calling ax.legend()
    handles, labels = ax.get_legend_handles_labels()

    fig.suptitle(title, fontsize=12, y=0.98)
    if handles:
        fig.legend(
            handles=handles,
            labels=labels,
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            ncol=n_cols,
            frameon=True,
        )

    fig.subplots_adjust(top=0.72)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_ranking_table(summary_rows: list[dict], show_loss: bool = True) -> None:
    """Print sorted ranking after all runs."""
    ranked = sorted(summary_rows, key=lambda r: (r["cv_late"], r["max_ratio_late"]))
    print("\n" + "=" * 88)
    print("ROUTING SWEEP RANKING (sorted by CV @ final step)")
    print("=" * 88)
    if show_loss:
        header = (
            f"{'Rank':<5} {'Config':<28} {'CV':>7} {'MaxRatio':>9} "
            f"{'DriftCV':>8} {'DriftMax':>9} {'Drop%':>6} {'Loss':>7}"
        )
    else:
        header = (
            f"{'Rank':<5} {'Config':<28} {'CV':>7} {'MaxRatio':>9} "
            f"{'DriftCV':>8} {'DriftMax':>9} {'Drop%':>6}"
        )
    print(header)
    print("-" * 88)
    for i, r in enumerate(ranked, start=1):
        if show_loss:
            print(
                f"{i:<5} {r['run_name']:<28} "
                f"{r['cv_late']:>7.3f} {r['max_ratio_late']:>9.2f} "
                f"{r['drift_cv']:>+8.3f} {r['drift_max']:>+9.3f} "
                f"{r['drop_pct_late']:>6.1f} {r['loss_late']:>7.2f}"
            )
        else:
            print(
                f"{i:<5} {r['run_name']:<28} "
                f"{r['cv_late']:>7.3f} {r['max_ratio_late']:>9.2f} "
                f"{r['drift_cv']:>+8.3f} {r['drift_max']:>+9.3f} "
                f"{r['drop_pct_late']:>6.1f}"
            )
    print("=" * 88 + "\n")