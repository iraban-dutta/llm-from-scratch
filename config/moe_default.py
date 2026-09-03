from config.constants import VOCAB_SIZE
from src.model.llm_config import LLMConfig
from src.training.trainer import LLMTrainerConfig


# ======== SET Global Configs ========
SEED = 42
B = 8
STEPS = 20000
DEVICE = 'auto'


# ======== DEFINE Model Config ========
llm_config = LLMConfig(
    vocab_size=VOCAB_SIZE,
    ctx_len=128,
    d_model=512,
    n_layer=8,
    ff_ratio=4,
    dropout=0.0,
    eps=1e-5,
    bias=False,
    position_embedding='sinusoidal',   # Bypassed to Identity if RoPE is used
    rotary_embedding=True,
    attention='gqa',
    normalization='layernorm',
    n_heads=8,
    n_groups=4,
    d_latent1=(512 // 8) * 2,
    d_latent2=(512 // 8) * 2,
    d_headR=(512 // 8) // 2,
    use_flash=True,
    attn_debug=False,
    use_moe=True,
    n_experts=4,
    n_shared_experts=0,
    topk=1,
    capacity_factor=1.25,
    use_vectorized_dispatch=True,
    noisy_router=False,
    router_noise_std=0.0,
    scale_aux_loss_expert_imp=0.0,
    scale_aux_loss_load_balance=0.0,
    aux_loss_free_load_balance=True,
    aux_loss_free_load_balance_bias_update=0.05,
)


# ======== DEFINE Trainer Config ========
trainer_config = LLMTrainerConfig(
    num_steps=STEPS,
    batch_size=B,
    learning_rate=6e-4,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.95,
    use_lr_scheduler=True,
    warmup_steps=15,
    min_lr=0.1*6e-4,
    grad_clip=1.0,
    log_interval=10,
    moe_log_interval=100,
    eval_interval=200,
    eval_steps=16,
    to_save_checkpoint=True,
    checkpoint_interval=1000,
    device=DEVICE,
)