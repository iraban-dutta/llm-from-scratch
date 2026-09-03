import argparse
import importlib
import torch
import json
from dataclasses import asdict
from pathlib import Path
from src.model.llm import LLM
from src.training.trainer import LLMTrainer
from main.run_utils import make_loaders

# ==============================================================================
# Save run config to log_dir/config.json
# ==============================================================================
def save_run_config(trainer, config_module=None, seed=None, resume_from=None):
    log_dir = Path(trainer.config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # On resume: keep config_module + seed from the first run's config.json
    old_json = log_dir / "config.json"
    if old_json.exists():
        with open(old_json) as f:
            old = json.load(f)
        if config_module is None:
            config_module = old.get("config_module")
        if seed is None:
            seed = old.get("seed")

    payload = {
        "config_module": config_module,
        "seed": seed,
        "resume_from": resume_from,
        "llm_config": asdict(trainer.model.config),
        "trainer_config": asdict(trainer.config),
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved run config: {log_dir / 'config.json'}")



# ==============================================================================
# Apply CLI overrides on top of trainer_config (config file or checkpoint)
# ==============================================================================
def apply_trainer_cli_overrides(trainer_config, args):
    if args.batch_size is not None:
        trainer_config.batch_size = args.batch_size
    if args.lr is not None:
        trainer_config.learning_rate = args.lr
    if args.log_interval is not None:
        trainer_config.log_interval = args.log_interval
    if args.eval_interval is not None:
        trainer_config.eval_interval = args.eval_interval
    if args.moe_log_interval is not None:
        trainer_config.moe_log_interval = args.moe_log_interval
    if args.checkpoint_interval is not None:
        trainer_config.checkpoint_interval = args.checkpoint_interval
    if args.to_save_checkpoint is not None:
        trainer_config.to_save_checkpoint = args.to_save_checkpoint


# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    default='config.dense_default',
    help='Config module for fresh training, e.g. config.dense_default',
)
parser.add_argument(
    '--resume',
    default=None,
    help='Path to checkpoint .pt — configs come from the checkpoint',
)
# Trainer overrides (optional — default None keeps config file / checkpoint value)
parser.add_argument('--batch-size', type=int, default=None)
parser.add_argument('--lr', type=float, default=None, help='Learning rate')
parser.add_argument('--log-interval', type=int, default=None)
parser.add_argument('--eval-interval', type=int, default=None)
parser.add_argument('--moe-log-interval', type=int, default=None)
parser.add_argument('--checkpoint-interval', type=int, default=None)
parser.add_argument(
    '--to-save-checkpoint',
    action=argparse.BooleanOptionalAction,
    default=None,
    help='Save checkpoints (use --no-to-save-checkpoint to disable)',
)
args = parser.parse_args()


# ==============================================================================
# Mode 1: Resume from checkpoint
# ==============================================================================
if args.resume:
    print(f'Resume training from checkpoint: {args.resume}')
    print('-' * 50)

    # Load checkpoint (args.resume: ckpt.pt path)
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)

    # Load model and trainer configs
    llm_config = ckpt['model_config']
    trainer_config = ckpt['train_config']

    # Apply CLI overrides on top of trainer_config (config file or checkpoint)
    print(f'Apply CLI overrides on top of trainer_config: {args}')
    apply_trainer_cli_overrides(trainer_config, args)
    print('-' * 50)

    print(llm_config)
    print('-' * 50)
    print(trainer_config)
    print('-' * 50)

    # Restore random state
    torch.set_rng_state(ckpt['rng_state'])

    # Create data loaders
    tok_bl, tok_bl_val = make_loaders(trainer_config, llm_config)
    tok_bl.curr_idx = ckpt['train_loader_state']['curr_idx']

    # Create model
    model = LLM(config=llm_config)

    # Load model state dict
    model.load_state_dict(ckpt['model_state_dict'])

    # Create trainer
    trainer = LLMTrainer(
        config=trainer_config,
        model=model,
        train_loader=tok_bl,
        val_loader=tok_bl_val,
    )
    # Load optimizer state dict
    trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    # Override learning rate if provided via CLI
    if args.lr is not None:
        for pg in trainer.optimizer.param_groups:
            pg['lr'] = args.lr
    # Load trainer state
    trainer.step = ckpt['step']
    # Load loss history
    trainer.train_loss_hist = ckpt['train_loss_hist']
    trainer.val_loss_hist = ckpt['val_loss_hist']
    # Load best validation loss
    if trainer.val_loss_hist:
        trainer.best_val_loss = min(trainer.val_loss_hist)

    # Save run config
    save_run_config(trainer, resume_from=args.resume)

    print(f'Resumed at step {trainer.step}')
    print('-' * 50)


# ==============================================================================
# Mode 2: Fresh training from config file
# ==============================================================================
else:
    print(f'Fresh training with config: {args.config}')
    print('-' * 50)

    # Import config from provided path (args.config: config.py path)
    cfg = importlib.import_module(args.config)

    # Set seed
    torch.manual_seed(cfg.SEED)

    # Define model and trainer configs
    llm_config = cfg.llm_config
    trainer_config = cfg.trainer_config

    # Apply CLI overrides on top of trainer_config (config file or checkpoint)
    print(f'Apply CLI overrides on top of trainer_config: {args}')
    apply_trainer_cli_overrides(trainer_config, args)
    print('-' * 50)

    print(llm_config)
    print('-' * 50)
    print(trainer_config)
    print('-' * 50)

    # Create data loaders
    tok_bl, tok_bl_val = make_loaders(trainer_config, llm_config)

    # Create model
    model = LLM(config=llm_config)

    # Create trainer
    trainer = LLMTrainer(
        config=trainer_config,
        model=model,
        train_loader=tok_bl,
        val_loader=tok_bl_val,
    )

    # Save run config
    save_run_config(trainer, config_module=args.config, seed=cfg.SEED)

# ==============================================================================
# Start Training
# ==============================================================================
print('Starting Training...')
print('-' * 50)
trainer.train()