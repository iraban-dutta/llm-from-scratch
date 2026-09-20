import time
import torch
import numpy as np
from src.model.llm_config import LLMConfig
from src.model.llm import LLM
from src.model.moe import MoE
from src.training.trainer import LLMTrainerConfig, LLMTrainer, TokenBatchLoader


# ========================================================
# Note:
# ========================================================
# Without MoE
#   We process B*ctx_len tokens per step (where B=8 and ctx_len=128).
# With MoE:
    # topk=1
        # We have n_experts=4, topk=1, capacity_factor=1 => same #tokens per batch wrt dense run, so no change in B (B=8 and ctx_len=128)
        # Perfect match with dense run in terms of #tokens processed per step.
        # But capacity_factor=1 limits #tokens per expert to (8*128*1*1)/4 = 256 tokens per expert (as n_experts=4)
        # Now if we have uneven routing, it might so happen that some experts are more busy than others, so we might drop tokens.
        # With capacity_factor=4, each expert can process upto 1024 tokens per step (8*128*1*4)/4
        # This is sufficient to ensure that no tokens are dropped.
        # Thus #tokens processed per step with MoE = #tokens processed per step with dense run = 8*128 = 1024 tokens per step.
    # topk=2
        # We have n_experts=4, topk=2, capacity_factor=1 => 2X more tokens per batch wrt dense run, so decrease B by 2X (B=4 and ctx_len=128)
        # Ensures that the #tokens seen by the MoE layer is matched with dense run (B=8 and ctx_len=128) - even though other layers see 2X less tokens
        # But capacity_factor=1 limits #tokens per expert to (4*128*2*1)/4 = 256 tokens per expert (as n_experts=4)
        # Now if we have uneven routing, it might so happen that some experts are more busy than others, so we might drop tokens.
        # With capacity_factor=4, each expert can process upto 1024 tokens per step (4*128*2*4)/4
        # This is sufficient to ensure that no tokens are dropped.
        # Thus #tokens processed per step with MoE = #tokens processed per step with dense run = 8*128 = 1024 tokens per step.


# ========================================================
# Benchmark Toggles
# ========================================================
USE_MOE = False                   # False → dense run (set B=8); True → MoE run (set B=8)
USE_VECTORIZED_DISPATCH = False   # MoE only: True → fast vectorized dispatch; False → legacy (torch.where) for TPUT comparison
RUN_PROFILER = False             # Step 6: set True to practice PyTorch profiler (forward only)
PRELOAD_BATCH = True             # Reuse one batch each step to reduce dataloader noise


TOPK = 1 
CAPACITY_FACTOR = 4  
assert TOPK in [1, 2], (
    f"TOPK must be 1 or 2, got {TOPK}"
)               
assert CAPACITY_FACTOR in [2, 4], (
    f"CAPACITY_FACTOR must be 2 or 4, got {CAPACITY_FACTOR}"
)


seed = 42
T = 128
B = 8                     # default batch size for dense run and MoE run with topk=1
if USE_MOE and TOPK==2:
    B = B//2               # batch size for MoE run with topk=2
WARMUP_STEPS = 300 #10
TRAIN_STEPS = 200 #20
DEVICE = 'auto'


# ==============================================================================
# Helpers
# ==============================================================================
def sync_device(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def ms_stats(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(np.median(arr)), float(np.mean(arr))


def pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole > 0 else 0.0


def patch_moe_forward_for_benchmark(model: torch.nn.Module) -> list:
    """MONKEY PATCHING: Temporarily patch MoE.forward to MoE.forward_benchmark."""
    patches = []
    for decoder in model.transformer.dec:
        if isinstance(decoder.mlp, MoE):
            patches.append((decoder.mlp, decoder.mlp.forward))
            decoder.mlp.forward = decoder.mlp.forward_benchmark
    return patches


def restore_moe_forward(patches: list) -> None:
    """RESTORE MONKEY PATCHING: Restore original MoE.forward from MoE.forward_benchmark."""
    for moe, original_forward in patches:
        moe.forward = original_forward


def register_mlp_hooks(model: torch.nn.Module, device: torch.device) -> tuple[list, list]:
    """Forward hooks on decoder.mlp (works for dense MLP and MoE)."""
    handles = []
    mlp_ms_per_step = []

    def pre_hook(module, inputs):
        sync_device(device)
        module._bench_t0 = time.perf_counter()

    def post_hook(module, inputs, output):
        sync_device(device)
        mlp_ms_per_step.append((time.perf_counter() - module._bench_t0) * 1000)

    for decoder in model.transformer.dec:
        handles.append(decoder.mlp.register_forward_pre_hook(pre_hook))
        handles.append(decoder.mlp.register_forward_hook(post_hook))

    return handles, mlp_ms_per_step


def remove_hooks(handles: list) -> None:
    """Remove all hooks from the model."""
    for h in handles:
        h.remove()


def collect_moe_bench_ms(model: torch.nn.Module) -> dict:
    """Collect MoE benchmark metrics from the model's last forward pass."""
    totals = {
        'route_ms': 0.0,
        'dispatch_ms': 0.0,
        'expert_ms': 0.0,
        'combine_ms': 0.0,
        'shared_ms': 0.0,
        'aux_ms': 0.0,
    }
    for decoder in model.transformer.dec:
        if isinstance(decoder.mlp, MoE):
            for key in totals:
                totals[key] += decoder.mlp.last_forward_bench_ms.get(key, 0.0)
    return totals


def moe_drop_rate(model: torch.nn.Module, batch_size: int, ctx_len: int, topk: int) -> float:
    """Compute the drop rate of MoE."""
    total_routed = batch_size * ctx_len * topk
    total_dropped = 0.0
    n_moe_layers = 0
    for decoder in model.transformer.dec:
        if isinstance(decoder.mlp, MoE):
            total_dropped += decoder.mlp.token_dropped.sum().item()
            n_moe_layers += 1
    if n_moe_layers == 0 or total_routed == 0:
        return 0.0
    return 100.0 * total_dropped / (total_routed * n_moe_layers)


def timed_train_step(
    trainer: LLMTrainer,
    x: torch.Tensor,
    y: torch.Tensor,
    mlp_ms_per_step: list,
) -> dict:
    """TIME TRAIN STEP: Time a single training step."""

    # Resolve device, model, and clear mlp_ms_per_step buffer.
    device = trainer.device
    model = trainer.model
    mlp_ms_per_step.clear()

    # Sync device, move tensors to device and record start time.
    sync_device(device)
    t0 = time.perf_counter()
    x = x.to(device)
    y = y.to(device)
    sync_device(device)
    t_data = time.perf_counter()

    # Zero gradients.
    trainer.optimizer.zero_grad()

    # Forward pass.
    sync_device(device)
    t_fwd_start = time.perf_counter()
    _, loss = model(x, y)
    sync_device(device)
    t_fwd_end = time.perf_counter()

    # Backward pass.
    sync_device(device)
    t_bwd_start = time.perf_counter()
    loss.backward()
    sync_device(device)
    t_bwd_end = time.perf_counter()

    # Optimizer step.
    sync_device(device)
    t_opt_start = time.perf_counter()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters=model.parameters(),
        max_norm=trainer.config.grad_clip,
        norm_type=2,
    )
    trainer.optimizer.step()
    sync_device(device)
    t_opt_end = time.perf_counter()

    # Record metrics.
    mlp_block_ms = float(sum(mlp_ms_per_step))

    # Collect metrics.
    record = {
        'data_ms': (t_data - t0) * 1000,
        'forward_ms': (t_fwd_end - t_fwd_start) * 1000,
        'backward_ms': (t_bwd_end - t_bwd_start) * 1000,
        'optim_ms': (t_opt_end - t_opt_start) * 1000,
        'step_ms': (t_opt_end - t0) * 1000,
        'mlp_block_ms': mlp_block_ms,
        'loss': loss.item(),
        'grad_norm': float(grad_norm),
    }

    # Collect MoE benchmark metrics if MoE is used.
    if model.config.use_moe:
        record['moe_bench_ms'] = collect_moe_bench_ms(model)
        record['drop_rate_pct'] = moe_drop_rate(
            model, trainer.config.batch_size, model.config.ctx_len, model.config.topk
        )

    return record


def print_summary(model: LLM, records: list[dict], batch_size: int) -> None:
    """Print the summary of the benchmark."""
    cfg = model.config
    ff_label = 'Forward-MoE' if cfg.use_moe else 'Forward-MLP'
    layer_tag = 'MoE' if cfg.use_moe else 'MLP'

    LW = 36
    W = 72

    n_input = batch_size * cfg.ctx_len

    if cfg.use_moe:
        drop_med, _ = ms_stats([r['drop_rate_pct'] for r in records])
        n_routed_1layer = n_input * cfg.topk
        n_routed_1layer_adj = n_routed_1layer * (1.0 - drop_med / 100.0)
        n_routed_all = n_routed_1layer * cfg.n_layer
    else:
        drop_med = 0.0
        n_routed_1layer = n_input
        n_routed_1layer_adj = n_input
        n_routed_all = n_input * cfg.n_layer

    def col(name: str) -> list[float]:
        return [r[name] for r in records]

    step_med, step_mean = ms_stats(col('step_ms'))
    data_med, _ = ms_stats(col('data_ms'))
    fwd_med, _ = ms_stats(col('forward_ms'))
    bwd_med, _ = ms_stats(col('backward_ms'))
    opt_med, _ = ms_stats(col('optim_ms'))
    mlp_med, _ = ms_stats(col('mlp_block_ms'))

    def row_ms(label: str, ms: float, pct_of: float) -> None:
        print(f"  {label:<{LW}} {ms:8.2f} ms  ({pct_of:5.1f}%)")

    def row_tok(label: str, tok_s: float) -> None:
        print(f"  {label:<{LW}} {tok_s:10,.0f} tok/s")

    def row_ms_scope(label: str, ms: float, pct_of: float, scope: str) -> None:
        print(f"  {label:<{LW}} {ms:8.2f} ms  ({pct_of:5.1f}% of {scope})")

    # ---- config ----
    print('\n' + '=' * W)
    print('BENCHMARK SUMMARY')
    print('=' * W)
    print('Config')
    print(f"  {'use_moe':<{LW}} {cfg.use_moe}")
    if cfg.use_moe:
        print(f"  {'moe settings':<{LW}} n_experts={cfg.n_experts}  topk={cfg.topk}  cap_factor={cfg.capacity_factor}")
        print(f"  {'vectorized dispatch':<{LW}} {cfg.use_vectorized_dispatch}")
    print(f"  {'batch':<{LW}} B={batch_size}  S={cfg.ctx_len}  layers={cfg.n_layer}")
    print(f"  {'input tokens/step':<{LW}} {n_input}  (B×S)")

    routed_1_label = f'routed tokens/step (1 {layer_tag} layer)'
    routed_all_label = f'routed tokens/step (all {layer_tag} layers)'
    formula_1 = 'B×S×topk' if cfg.use_moe else 'B×S'
    formula_all = 'B×S×topk×layers' if cfg.use_moe else 'B×S×layers'
    print(f"  {routed_1_label:<{LW}} {n_routed_1layer}  ({formula_1})")
    print(f"  {routed_all_label:<{LW}} {n_routed_all:.0f}  ({formula_all})")    
    print(f"  {'steps':<{LW}} {len(records)}")
    print(f"  {'step time':<{LW}} {step_med:.2f} ms median  ({step_mean:.2f} ms mean)")

    # ---- train step ----
    print('-' * W)
    print('Train step breakdown (median)')
    row_ms('Data load', data_med, pct(data_med, step_med))
    row_ms('Forward', fwd_med, pct(fwd_med, step_med))
    row_ms('Backward', bwd_med, pct(bwd_med, step_med))
    row_ms('Optim+clip', opt_med, pct(opt_med, step_med))
    row_ms('Total', step_med, 100.0)

    # ---- forward breakdown ----
    rest_fwd_med = max(0.0, fwd_med - mlp_med)
    print('-' * W)
    print('Forward breakdown (median)')
    row_ms_scope(ff_label, mlp_med, pct(mlp_med, fwd_med), 'forward')
    row_ms_scope('Rest (attn+emb+head)', rest_fwd_med, pct(rest_fwd_med, fwd_med), 'forward')
    row_ms_scope('Total forward', fwd_med, 100.0, 'forward')

    # ---- MoE internals ----
    if cfg.use_moe:
        keys = ['route_ms', 'dispatch_ms', 'expert_ms', 'combine_ms', 'shared_ms', 'aux_ms']
        labels = ['Route', 'Dispatch', 'Expert GEMM', 'Combine', 'Shared experts', 'Aux losses']
        moe_totals = {k: [r['moe_bench_ms'][k] for r in records] for k in keys}
        moe_med = {k: ms_stats(moe_totals[k])[0] for k in keys}
        moe_sum = sum(moe_med[k] for k in keys)

        print('-' * W)
        print('MoE forward internals (median ms/step, summed over all MoE layers)')
        for key, label in zip(keys, labels):
            if key == 'shared_ms' and moe_med[key] == 0:
                continue
            row_ms_scope(label, moe_med[key], pct(moe_med[key], moe_sum), 'MoE fwd')
        row_ms_scope('Total forward', moe_sum, 100.0, 'MoE fwd')
        print(f"  {'Hook total':<{LW}} {mlp_med:8.2f} ms  (Total forward ~ Hook total)")

    # ---- throughput ----
    print('-' * W)
    print(f'TPUT  [numerator = B×S = {n_input}]')
    row_tok('Complete', n_input / (step_med / 1000))
    row_tok('Forward', n_input / (fwd_med / 1000))
    row_tok('Backward', n_input / (bwd_med / 1000))
    row_tok('Optim+clip', n_input / (opt_med / 1000))

    if not cfg.use_moe:
        row_tok('Forward-MLP', n_input / (mlp_med / 1000))
    else:
        print('-' * W)
        print(f'Forward-MoE  [numerator = B×S×topk = {n_routed_1layer}; each token → topk={cfg.topk} experts/layer]')
        row_tok('Nominal', n_routed_1layer / (mlp_med / 1000))
        row_tok('Drop-adj', n_routed_1layer_adj / (mlp_med / 1000))

    print('=' * W + '\n')


def run_profiler(trainer: LLMTrainer, x: torch.Tensor, y: torch.Tensor, steps: int = 5) -> None:
    """Forward-only profiler practice block (Step 6)."""
    device = trainer.device
    model = trainer.model
    x, y = x.to(device), y.to(device)
    moe_patches = patch_moe_forward_for_benchmark(model) if model.config.use_moe else []

    print('\nRunning PyTorch profiler (forward only)...')
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            for _ in range(steps):
                model(x, y)
                sync_device(device)

        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    finally:
        restore_moe_forward(moe_patches)


# ==============================================================================
# Start of Script
# ==============================================================================


# ======== SET SEED ========
torch.manual_seed(seed)

# ======== DEFINE Model Config ========
llm_config = LLMConfig(
    vocab_size=50304,
    ctx_len=T,
    d_model=512,
    n_layer=8,
    ff_ratio=4,
    dropout=0.0,
    eps=1e-5,
    bias=False,
    position_embedding='sinusoidal',
    rotary_embedding=False,
    attention='gqa',
    normalization='layernorm',
    n_heads=8,
    n_groups=4,
    d_latent1=(512 // 8) * 2,
    d_latent2=(512 // 8) * 2,
    d_headR=(512 // 8) // 2,
    use_flash=True,
    attn_debug=False,
    use_moe=USE_MOE,
    n_experts=4,
    n_shared_experts=0,
    topk=TOPK,
    capacity_factor=CAPACITY_FACTOR,
    use_vectorized_dispatch=USE_VECTORIZED_DISPATCH,
    noisy_router=False,
    router_noise_std=0.0,
    scale_aux_loss_expert_imp=0.0,
    scale_aux_loss_load_balance=0.0,
    aux_loss_free_load_balance=True,
    aux_loss_free_load_balance_bias_update=0.05,
)

# ======== DEFINE Trainer Config ========
trainer_config = LLMTrainerConfig(
    num_steps=TRAIN_STEPS,
    batch_size=B,
    learning_rate=3e-4,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.95,
    use_lr_scheduler=False,
    warmup_steps=15,
    min_lr=0.1 * 3e-4,
    grad_clip=1.0,
    log_interval=5,
    moe_log_interval=50,
    eval_interval=50,
    eval_steps=32,
    to_save_checkpoint=False,
    checkpoint_interval=50,
    device=DEVICE,
)

# ======== Print Configs ========
print(llm_config)
print(trainer_config)
print('-' * 50)

# ======== Instantiate Train Batch Loaders ========
tok_bl = TokenBatchLoader(
    B=trainer_config.batch_size,
    T=llm_config.ctx_len,
    binary_file_path='./data/tinystories/processed/train.bin',
    dtype=np.uint16,
    debug=False,
)

# ======== Instantiate Model and Trainer ========
model = LLM(config=llm_config)
trainer = LLMTrainer(
    config=trainer_config,
    model=model,
    train_loader=tok_bl,
    val_loader=None,
)

# ======== Set Model to Training Mode ========
trainer.model.train()

# ======== Patch MoE.forward for Benchmarking ========
moe_patches = patch_moe_forward_for_benchmark(trainer.model) if USE_MOE else []

# ======== Register MLP Hooks for Timing ========
hook_handles, mlp_ms_buffer = register_mlp_hooks(trainer.model, trainer.device)

# ======== Get Fixed Batch for Stable Timing (Controlled by PRELOAD_BATCH) ========
fixed_x, fixed_y = trainer.train_loader.next_batch()

# ======== Warmup ========
print('Warmup...')
for i in range(WARMUP_STEPS):
    print(f'Running WARMUP step: {i+1}')
    x, y = (fixed_x, fixed_y) if PRELOAD_BATCH else trainer.train_loader.next_batch()
    timed_train_step(trainer, x, y, mlp_ms_buffer)
sync_device(trainer.device)
print('Warmup done.')
print('-' * 50)

# ======== Train ========
records = []
for i in range(TRAIN_STEPS):
    x, y = (fixed_x, fixed_y) if PRELOAD_BATCH else trainer.train_loader.next_batch()
    records.append(timed_train_step(trainer, x, y, mlp_ms_buffer))
    if (i + 1) % 20 == 0:
        med, _ = ms_stats([r['step_ms'] for r in records])
        print(f"  step {i + 1}/{TRAIN_STEPS} | median step so far: {med:.2f} ms")

# ======== Remove MLP Hooks and Restore MoE.forward ========
remove_hooks(hook_handles)
restore_moe_forward(moe_patches)

# ======== Print Summary ========
print_summary(trainer.model, records, B)

# ======== Run Profiler (Optional) ========
if RUN_PROFILER:
    run_profiler(trainer, fixed_x, fixed_y)