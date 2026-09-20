"""
MoE routing balance tuner.

Phase 1: short sweep (~200 steps) over baseline / noisy / loss-free / both.
Phase 2: set PHASE=2 and paste top configs into PHASE2_CONFIGS for longer runs.

Usage:
  python main/tune_moe_routing.py
  python main/tune_moe_routing.py --analyze-only logs/moe_routing_tune/<timestamp>
"""

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

from src.model.llm_config import LLMConfig
from src.model.llm import LLM
from src.training.trainer import LLMTrainerConfig, LLMTrainer, TokenBatchLoader
from src.training.moe_routing_metrics import (
    analyze_run,
    print_run_report,
    print_ranking_table,
    plot_layer_token_distr_band,
    plot_config_comparison,
    plot_metric_over_steps,
    fair_share,
)


# ==============================================================================
# Phase control
# ==============================================================================
PHASE = 1                     # 1 = short sweep, 2 = validate top configs
TRAIN_STEPS = 200             # use 500 when PHASE=2
MOE_LOG_INTERVAL = 10         # moe_stats.csv snapshot every N steps
SEED = 42

# Phase 2 only: paste top configs from phase 1 ranking here
PHASE2_CONFIGS = [
    {"name": "lossfree_0.01",    "noise_std": 0.0,  "bias_update": 1e-2},
    {"name": "both_0.15_0.01",   "noise_std": 0.15, "bias_update": 1e-2},
    {"name": "lossfree_0.005",   "noise_std": 0.0,  "bias_update": 5e-3}
]


# ==============================================================================
# Model architecture (train_benchmark_detailed + RoPE)
# ==============================================================================
T = 128 
B = 8                         # MoE: 2x routed tokens vs dense B=16
N_LAYER = 8 
D_MODEL = 512
FF_RATIO = 4

# Attention: 'gqa' or 'mhla'
ATTENTION = "gqa"
ROTARY = True
N_HEADS = 8
N_GROUPS = 4                  # GQA
D_LATENT1 = (512 // 8) * 2    # MHLA
D_LATENT2 = (512 // 8) * 2
D_HEADR = (512 // 8) // 2

# MoE architecture
N_EXPERTS = 4
TOPK = 1
CAPACITY_FACTOR = 1.25
N_SHARED_EXPERTS = 0

# Data
TRAIN_BIN = "./data/tinystories/processed/train.bin"
VAL_BIN = "./data/tinystories/processed/val.bin"


# ==============================================================================
# Sweep grids (no aux-loss terms — only noisy router and/or loss-free LB)
# ==============================================================================

# Point sweep configs:
SWEEP_CONFIGS = [
    # reference
    {"name": "baseline",           "noise_std": 0.0,  "bias_update": 0.0},
    # one noisy-only sanity check (expect: still bad, confirms loss-free is needed)
    # {"name": "noisy_0.1",          "noise_std": 0.1,  "bias_update": 0.0},
    # {"name": "noisy_0.2",          "noise_std": 0.2,  "bias_update": 0.0},
    # loss-free only (main sweep)
    {"name": "lossfree_0.001",     "noise_std": 0.0,  "bias_update": 1e-3},
    {"name": "lossfree_0.005",     "noise_std": 0.0,  "bias_update": 5e-3},
    {"name": "lossfree_0.01",      "noise_std": 0.0,  "bias_update": 1e-2},
    {"name": "lossfree_0.05",      "noise_std": 0.0,  "bias_update": 5e-2},
    # both (spot-check: does noise help once loss-free is on?)
    # {"name": "both_0.15_0.005",     "noise_std": 0.15,  "bias_update": 5e-3},
    # {"name": "both_0.15_0.01",      "noise_std": 0.15,  "bias_update": 1e-2},
]

# Matrix sweep configs:
# NOISE_STD_GRID = [0.01, 0.05, 0.1, 0.5]
# BIAS_UPDATE_GRID = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]

# ==============================================================================
# Helpers
# ==============================================================================

# ----------------------------------------------------------------
# Build sweep configs for point sweep.
# ----------------------------------------------------------------
def build_sweep_configs() -> list[dict]:
    """Return configs for this phase."""
    if PHASE == 2:
        return PHASE2_CONFIGS
    return SWEEP_CONFIGS

# ----------------------------------------------------------------
# Build sweep configs for matrix sweep.
# ----------------------------------------------------------------
# def build_sweep_configs() -> list[dict]:
#     """
#     Build list of routing configs to try.
#     Families: baseline, noisy-only, loss-free-only, both.
#     """
#     if PHASE == 2:
#         return PHASE2_CONFIGS

#     configs = [
#         {"name": "baseline", "noise_std": 0.0, "bias_update": 0.0},
#     ]

#     for noise in NOISE_STD_GRID:
#         configs.append({"name": f"noisy_{noise}", "noise_std": noise, "bias_update": 0.0})

#     for bias in BIAS_UPDATE_GRID:
#         configs.append({"name": f"lossfree_{bias}", "noise_std": 0.0, "bias_update": bias})

#     # Small combo grid (noisy x loss-free) — skip if you want fewer runs
#     for noise in [0.05, 0.1]:
#         for bias in [1e-3, 5e-3, 1e-2]:
#             configs.append({
#                 "name": f"both_{noise}_{bias}",
#                 "noise_std": noise,
#                 "bias_update": bias,
#             })

#     return configs


def checkpoint_steps() -> list[int]:
    """Steps where we report aggregated metrics."""
    if TRAIN_STEPS >= 500:
        return [50, 100, 200, 500]
    return [50, 100, 200]


def build_llm_config(noise_std: float, bias_update: float) -> LLMConfig:
    """Model config: architecture fixed, routing knobs vary."""
    return LLMConfig(
        vocab_size=50304,
        ctx_len=T,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        ff_ratio=FF_RATIO,
        dropout=0.0,
        eps=1e-5,
        bias=False,
        position_embedding="sinusoidal",
        rotary_embedding=ROTARY,
        attention=ATTENTION,
        normalization="layernorm",
        n_heads=N_HEADS,
        n_groups=N_GROUPS,
        d_latent1=D_LATENT1,
        d_latent2=D_LATENT2,
        d_headR=D_HEADR,
        use_flash=True,
        attn_debug=False,
        use_moe=True,
        n_experts=N_EXPERTS,
        n_shared_experts=N_SHARED_EXPERTS,
        topk=TOPK,
        capacity_factor=CAPACITY_FACTOR,
        use_vectorized_dispatch=True,
        # --- routing knobs (the only things we sweep) ---
        noisy_router=(noise_std > 0),
        router_noise_std=noise_std,
        scale_aux_loss_expert_imp=0.0,
        scale_aux_loss_load_balance=0.0,
        aux_loss_free_load_balance=(bias_update > 0),
        aux_loss_free_load_balance_bias_update=bias_update,
    )


def run_one_config(cfg: dict, run_dir: Path) -> dict:
    """Train one routing config and return metrics."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    llm_config = build_llm_config(cfg["noise_std"], cfg["bias_update"])
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "run_name": cfg["name"],
            "noise_std": cfg["noise_std"],
            "bias_update": cfg["bias_update"],
            "train_steps": TRAIN_STEPS,
            "llm_config": asdict(llm_config),
        }, f, indent=2)

    # Trainer writes moe_stats.csv into log_dir
    trainer_config = LLMTrainerConfig(
        num_steps=TRAIN_STEPS,
        batch_size=B,
        learning_rate=3e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        use_lr_scheduler=False,
        grad_clip=1.0,
        log_interval=50,
        log_dir=str(run_dir),
        moe_log_interval=MOE_LOG_INTERVAL,
        eval_interval=TRAIN_STEPS + 999,   # skip mid-training eval (faster)
        eval_steps=16,
        to_save_checkpoint=False,
        device="auto",
    )

    # Fresh batches every step (train_loader.curr_idx advances naturally)
    train_loader = TokenBatchLoader(
        B=B, T=T, binary_file_path=TRAIN_BIN, dtype=np.uint16, debug=False,
    )
    val_loader = TokenBatchLoader(
        B=B, T=T, binary_file_path=VAL_BIN, dtype=np.uint16, debug=False,
    )

    torch.manual_seed(SEED)
    model = LLM(config=llm_config)
    trainer = LLMTrainer(
        config=trainer_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print(f"\n>>> Training: {cfg['name']}  ({TRAIN_STEPS} steps)")
    trainer.train()

    loss_late = trainer.train_loss_hist[-1] if trainer.train_loss_hist else float("nan")

    # Analyze moe_stats.csv
    csv_path = run_dir / "moe_stats.csv"
    ckpt_steps = checkpoint_steps()
    metrics = analyze_run(
        csv_path=csv_path,
        n_experts=N_EXPERTS,
        n_layers=N_LAYER,
        batch_size=B,
        ctx_len=T,
        topk=TOPK,
        checkpoint_steps=ckpt_steps,
    )

    print_run_report(cfg["name"], metrics, ckpt_steps, N_EXPERTS)

    # Per-run layer band plot
    plot_layer_token_distr_band(
        csv_path=csv_path,
        n_experts=N_EXPERTS,
        save_path=run_dir / "layer_token_distr_band.png",
        target=fair_share(N_EXPERTS),
    )

    late_step = ckpt_steps[-1]
    late = metrics["checkpoints"].get(late_step, {})

    return {
        "run_name": cfg["name"],
        "noise_std": cfg["noise_std"],
        "bias_update": cfg["bias_update"],
        "cv_late": late.get("cv", float("nan")),
        "max_ratio_late": late.get("max_load_ratio", float("nan")),
        "drift_cv": metrics["drift_cv"],
        "drift_max": metrics["drift_max"],
        "drop_pct_late": metrics["drop_pct_late"],
        "loss_late": loss_late,
        "steps": metrics["steps"],
        "cv_series": metrics["cv_series"],
        "max_load_series": metrics["max_load_series"],
        "run_dir": str(run_dir),
    }


def save_sweep_summary(sweep_dir: Path, summary_rows: list[dict]) -> None:
    """Write sweep_summary.csv (aggregated metrics only)."""
    path = sweep_dir / "sweep_summary.csv"
    fieldnames = [
        "run_name", "noise_std", "bias_update",
        "cv_late", "max_ratio_late", "drift_cv", "drift_max",
        "drop_pct_late", "loss_late", "run_dir",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row[k] for k in fieldnames})


def analyze_existing_sweep(sweep_dir: Path) -> None:
    """Re-analyze + replot from an existing sweep folder (no training)."""
    summary_rows = []
    ckpt_steps = checkpoint_steps()

    for run_dir in sorted(sweep_dir.glob("run_*")):
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)

        csv_path = run_dir / "moe_stats.csv"
        if not csv_path.exists():
            print(f"Skipping {run_dir.name}: no moe_stats.csv")
            continue

        metrics = analyze_run(
            csv_path=csv_path,
            n_experts=N_EXPERTS,
            n_layers=N_LAYER,
            batch_size=B,
            ctx_len=T,
            topk=TOPK,
            checkpoint_steps=ckpt_steps,
        )
        print_run_report(cfg["run_name"], metrics, ckpt_steps, N_EXPERTS)

        plot_layer_token_distr_band(
            csv_path=csv_path,
            n_experts=N_EXPERTS,
            save_path=run_dir / "layer_token_distr_band.png",
        )

        late_step = ckpt_steps[-1]
        late = metrics["checkpoints"].get(late_step, {})
        summary_rows.append({
            "run_name": cfg["run_name"],
            "noise_std": cfg["noise_std"],
            "bias_update": cfg["bias_update"],
            "cv_late": late.get("cv", float("nan")),
            "max_ratio_late": late.get("max_load_ratio", float("nan")),
            "drift_cv": metrics["drift_cv"],
            "drift_max": metrics["drift_max"],
            "drop_pct_late": metrics["drop_pct_late"],
            "loss_late": float("nan"),
            "steps": metrics["steps"],
            "cv_series": metrics["cv_series"],
            "max_load_series": metrics["max_load_series"],
            "run_dir": str(run_dir),
        })

    _make_global_plots(sweep_dir, summary_rows, show_loss=False)


def _make_global_plots(sweep_dir: Path, summary_rows: list[dict], show_loss: bool = True) -> None:
    """Global comparison plots across all configs."""
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_sweep_summary(sweep_dir, summary_rows)
    print_ranking_table(summary_rows, show_loss=show_loss)

    plot_config_comparison(summary_rows, plots_dir / "config_comparison.png")

    plot_metric_over_steps(
        all_runs=summary_rows,
        metric_key="cv_series",
        ylabel="CV (avg over layers)",
        title="Routing CV over steps",
        save_path=plots_dir / "cv_over_steps.png",
    )

    plot_metric_over_steps(
        all_runs=summary_rows,
        metric_key="max_load_series",
        ylabel="Max load ratio (avg over layers)",
        title="Max load ratio over steps",
        save_path=plots_dir / "max_load_over_steps.png",
        ideal_line=1.0,
    )

    print(f"Plots saved to: {plots_dir}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analyze-only",
        type=str,
        default=None,
        help="Replot from existing sweep dir, e.g. logs/moe_routing_tune/2026_09_01_12_00_00",
    )
    args = parser.parse_args()

    if args.analyze_only:
        analyze_existing_sweep(Path(args.analyze_only))
        return

    # Output root for this sweep
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    sweep_dir = Path(f"./logs/moe_routing_tune/{timestamp}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("MoE ROUTING TUNE")
    print("=" * 64)
    print(f"  phase={PHASE}  steps={TRAIN_STEPS}  attention={ATTENTION}  rotary={ROTARY}")
    print(f"  n_experts={N_EXPERTS}  topk={TOPK}  capacity={CAPACITY_FACTOR}")
    print(f"  output: {sweep_dir}")

    sweep_configs = build_sweep_configs()
    print(f"  configs to run: {len(sweep_configs)}")
    print("=" * 64)

    summary_rows = []
    for cfg in sweep_configs:
        run_dir = sweep_dir / f"run_{cfg['name']}"
        row = run_one_config(cfg, run_dir)
        summary_rows.append(row)

    _make_global_plots(sweep_dir, summary_rows)


if __name__ == "__main__":
    main()