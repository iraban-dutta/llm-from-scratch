# ISSUE: Low Batch size (B=4) 
# Triggering train() on mps sometimes is causing grad_norm to go high
# Behaviour is non-deterministic - hard to reproduce, Increasing Batch Size makes it a bit more stable

import time
from pathlib import Path
from typing import Any
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Tuple
from src.model.llm_config import LLMConfig
from src.model.llm import LLM



class TokenBatchLoader:
    def __init__(self, B:int, T:int, binary_file_path:str, dtype:np.dtype, debug:bool=False):
        self.B = max(1, B)
        self.T = max(4, T)
        if not debug:
            self.tokens = np.memmap(
                filename=binary_file_path,
                dtype=dtype,
                mode='r'
            )
            print(f"Loaded {len(self.tokens)/1e6}M tokens") 
            print(f"1 epoch ~ {len(self.tokens)//(self.B*self.T)} steps")
            print('-'*50)
        else:
            self.tokens = np.arange(1, 101)
        if len(self.tokens)<=(self.B)*(self.T):
            raise ValueError(
                f"Current values of batch size {B} and seq length {T} are too large for dataset, please reduce either or both"
            )
        self.curr_idx = 0

    def next_batch(self) -> Tuple[torch.tensor]:
        B, T = (self.B), (self.T)

        buffer = self.tokens[self.curr_idx:self.curr_idx+(B*T+1)]
        x = torch.tensor(buffer[:-1]).view(B, T)
        y = torch.tensor(buffer[1:]).view(B, T)

        # Covert x,y from uint16 to int32 and int64
        x = x.int()
        y = y.long()

        self.curr_idx += B*T
        if self.curr_idx+(B*T+1) > len(self.tokens):
            self.curr_idx=0

        return x, y



@dataclass
class LLMTrainerConfig:

    # Training Steps:
    num_steps:int=50

    # Batch
    batch_size:int=32

    # Optimizer
    learning_rate:float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95

    # LR Scheduler
    use_lr_scheduler: bool = False
    warmup_steps: int = 15
    min_lr: float = 0.1*3e-4

    # Gradient clipping
    grad_clip: float = 1.0

    # Logging
    curr_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_interval: int = 10
    log_dir: str = f"./logs/{curr_datetime}"
    moe_log_interval: int = 100

    # Evaluation
    eval_interval: int = 100
    eval_steps: int    = 16*32 # (16X of train batch size)

    # Checkpointing
    to_save_checkpoint: bool = False
    checkpoint_interval: int = 500
    checkpoint_dir: str = f"./checkpoints/{curr_datetime}"

    # Device
    device: str = "auto"

    # Early stopping
    # early_stopping_patience: int | None = None
    

class LLMTrainer:
    def __init__(
        self, 
        config:LLMTrainerConfig, 
        model:nn.Module,
        train_loader:TokenBatchLoader,
        val_loader:TokenBatchLoader | None
    ):
        # Traing Configs/Hyperparams
        self.config=config
        # Detect device 
        self.device = self._resolve_device()
        # Move model to device
        self.model = model.to(self.device)
        # Define object to load batches of tokens of shape (B, T)
        self.train_loader=train_loader
        self.val_loader=val_loader
        # Define Optimizer
        self.optimizer = self._configure_optimizer()
        # Track current step of update
        self.step=0
        # Track best val loss
        self.best_val_loss = torch.inf
        # Track Loss over steps
        self.train_loss_hist = []
        self.val_loss_hist   = []
        # Initialize train.log file
        log_path = Path(self.config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self.log_file_path = log_path/'train.log'
        # Initialize moe_stats.csv file
        self.moe_stats_file_path = log_path/'moe_stats.csv'
        if (self.model.config.use_moe) and not (self.moe_stats_file_path.exists()):
            with open(self.moe_stats_file_path, "w") as f:
                f.write(
                    "step,layer,expert,token_distr,expert_imp,"
                    "token_dropped,expert_bias\n"
                )


    def _resolve_device(self) -> torch.device:

        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')


    def _configure_optimizer(self):
        param_dict = {pn:p for pn, p in self.model.named_parameters() if p.requires_grad==True}

        decay_params = [p for pn, p in param_dict.items() if p.ndim>=2]
        non_decay_params = [p for pn, p in param_dict.items() if p.ndim<2]
        num_decay_params = sum(p.nelement() for p in decay_params)
        num_non_decay_params = sum(p.nelement() for p in non_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(non_decay_params)}, with {num_non_decay_params:,} parameters")
        per_param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]

        # Define optimizer
        # use 'fused' if device is cuda: Supports a fused kernel for all parameter updates
        use_fused = self.device.type == "cuda"
        optimizer = AdamW(
            per_param_groups, 
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            fused=use_fused
        )

        return optimizer


    def evaluate_model(self) -> float:

        # Dont track grads for eval
        with torch.no_grad():

            # For evaluating over the same #eval_steps batches - Comment out
            self.val_loader.curr_idx = 0

            # Set model to eval mode
            self.model.eval()

            cnt, val_loss = 0, 0.0
            while cnt<self.config.eval_steps:
                # Load Data (x,y) and move tensors to device
                x_val, y_val = self.val_loader.next_batch()
                x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                _, loss = self.model(x_val, y_val)
                val_loss += loss.item()
                cnt += 1

            # Reset model to training mode
            self.model.train()

        return (val_loss/self.config.eval_steps)


    def _get_lr(self, init_lr:float, final_lr:float, max_step:int, warmup_steps:int, curr_step:int):
        
        assert final_lr>0 and 0<(final_lr/init_lr)<1, (
            f"LRs should be > 0. Also final LR should be less than inital LR, got values: initial LR: {init_lr} & final LR: {final_lr}"
        )
        warmup_steps_reslv = min(warmup_steps, int(0.05*max_step))
        if curr_step<warmup_steps_reslv:
            # Linear Warmup
            return init_lr*((curr_step+1)/warmup_steps_reslv)
        elif curr_step>max_step:
            # Keep training with final_lr
            return final_lr

        # Cosine Decay (From warmup_step to max_step)
        cosine_curve = 0.5*(np.cos((np.pi/(max_step-warmup_steps_reslv))*(curr_step-warmup_steps_reslv)) + 1)
        return (init_lr-final_lr)*cosine_curve + final_lr


    def train_step(self):

            # Load Data (x,y) and move tensors to device
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)

            # Zero out gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, loss = self.model(x, y)

            # Backward Pass
            loss.backward()

            # Clip gardient norm
            # Computes the global norm across all parameter grads
            # if ||g|| > max_norm, each grad is scaled: g_i = g_i * max_norm / ||g|| 
            # The function returns the PRE-clipping global norm.
            global_grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config.grad_clip, norm_type=2)

            # DEBUG: Grad norm without clipping
            # global_grad_norm = sum(torch.linalg.norm(p)**2 for p in self.model.parameters())**0.5

            # Added checks to raise RuntimeError or a warning for unsually large grad_norms
            if not torch.isfinite(global_grad_norm):
                raise RuntimeError(
                    f"Non-finite gradient norm at step {self.step}: "
                    f"{global_grad_norm.item()}"
                )

            if global_grad_norm > 100:
                print(
                    f"WARNING: unusually large gradient norm "
                    f"{global_grad_norm.item():.2f} at step {self.step}"
                )

            # Optimizer Step
            self.optimizer.step()



            return loss.item(), global_grad_norm.item()


    def train(self):

        val_loss = 0.0
        # Train Loop
        while self.step < self.config.num_steps:

            if self.config.use_lr_scheduler:
                # Set LR for current step
                lr = self._get_lr(
                    init_lr=self.config.learning_rate, 
                    final_lr=self.config.min_lr, 
                    max_step=self.config.num_steps, 
                    warmup_steps=self.config.warmup_steps, 
                    curr_step=self.step)

                # Set lr in optimizer
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr
            else:
                lr = self.config.learning_rate

            # Start time
            step_start = time.perf_counter()

            # Perfom 1 unit of train step
            train_loss, grad_norm = self.train_step()
            self.train_loss_hist.append(train_loss)

            # Synchronize
            if self.device.type == 'mps':
                torch.mps.synchronize()

            # End time
            step_end = time.perf_counter()

            step_total_time = (step_end-step_start)
            train_throughput = (self.config.batch_size*self.model.config.ctx_len)/step_total_time


            # Logging: Log Train Stats @ eval_interval or @ log_iterval
            if self.step % self.config.eval_interval == 0:
                val_loss = self.evaluate_model()
                self.val_loss_hist.append(val_loss)
                self._log(f"Step: {self.step}, BatchTime: {step_total_time*1000:.2f} ms, TPUT: {train_throughput:.2f} tok/sec, LR: {lr:.6f}, GradNorm: {grad_norm:.2f}, Loss: {train_loss:.4f}, Val_Loss: {val_loss:.4f}")
                # Best Val loss obtained
                if self.config.to_save_checkpoint and val_loss < self.best_val_loss:
                    self.save_checkpoint(filename='best.pt') # best.pt saved
                    self.best_val_loss = val_loss
            elif self.step % self.config.log_interval == 0:
                # Track normal train stats @ every log_iterval
                self._log(f"Step: {self.step}, BatchTime: {step_total_time*1000:.2f} ms, TPUT: {train_throughput:.2f} tok/sec, LR: {lr:.6f}, GradNorm: {grad_norm:.2f}, Loss: {train_loss:.4f}")

            # MoE Logging: Log MoE Routing stats @ every moe_log_iterval
            if (self.model.config.use_moe) and (self.step % self.config.moe_log_interval == 0):
                self._log_moe_stats()
                self._save_moe_stats()
                   
            # Checkpointing
            if self.config.to_save_checkpoint and self.step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(filename='latest.pt') # latest.pt saved

            # Update step attribute    
            self.step += 1

        # Final Logging: Log Train Stats @ eval_interval or @ log_iterval
        val_loss = self.evaluate_model()
        self.val_loss_hist.append(val_loss)
        self._log(f"Step: {self.step}, BatchTime: {step_total_time*1000:.2f} ms, TPUT: {train_throughput:.2f} tok/sec, LR: {lr:.6f}, GradNorm: {grad_norm:.2f}, Loss: {train_loss:.4f}, Val_Loss: {val_loss:.4f}")
        # Best Val loss obtained
        if self.config.to_save_checkpoint and val_loss < self.best_val_loss:
            self.save_checkpoint(filename='best.pt') # best.pt saved
            self.best_val_loss = val_loss

        # Final MoE Logging: Log MoE Routing stats @ every moe_log_iterval
        if self.model.config.use_moe:
            self._log_moe_stats()
            self._save_moe_stats()

        
    def save_checkpoint(self, filename:str='latest.pt') -> None:

        ckpt_path = Path(self.config.checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_path/filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model.config,
            "train_config": self.config,
            "step": self.step,                             # Num of update steps while training
            "train_loss_hist": self.train_loss_hist,       # Train loss dumped for every step 
            "val_loss_hist": self.val_loss_hist,           # Val loss dumped for every step 
            "train_loader_state": {
                "curr_idx": self.train_loader.curr_idx,    # Store the curr_idx of train_loader - so that we resume form the same batch
            },
            "rng_state": torch.get_rng_state(),            # While loading: torch.set_rng_state(checkpoint["rng_state"])
        }

        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
        

    def load_checkpoint(self, file_path:str, weights_only:bool=False) -> Any:
        file_path = Path(file_path)
        if file_path.exists():
            return torch.load(file_path, weights_only=weights_only)
        else:
            raise FileNotFoundError(
                f"File not found at path {str(file_path)}"
            )


    def _log(self, msg: str) -> None:
        print(msg)
        with open(self.log_file_path, "a") as f:
            f.write(msg + "\n")


    def _log_moe_stats(self) -> None:
        self._log("MoE Routing Stats:")
        for layer_idx, decoder in enumerate(self.model.transformer.dec):
            moe = decoder.mlp

            token_distr = moe.token_distr.detach().cpu().numpy()

            # Use fewer summary statistics for small numbers of experts.
            if self.model.config.n_experts <= 4:
                token_dist_summary = (
                    f"min={np.min(token_distr):.3f} "
                    f"p50={np.percentile(token_distr, 50):.3f} "
                    f"max={np.max(token_distr):.3f}"
                )
            else:
                token_dist_summary = (
                    f"min={np.min(token_distr):.3f} "
                    f"p25={np.percentile(token_distr, 25):.3f} "
                    f"p50={np.percentile(token_distr, 50):.3f} "
                    f"p75={np.percentile(token_distr, 75):.3f} "
                    f"max={np.max(token_distr):.3f}"
                )


            dropped_pct = 100.0 * (moe.token_dropped.sum().item() / (self.config.batch_size * self.model.config.ctx_len * self.model.config.topk))

            self._log(
                f"Layer: {layer_idx} | "
                f"TokenDist: {token_dist_summary} | "
                f"Dropped: {dropped_pct:.2f}%"
            )

    def _save_moe_stats(self) -> None:
        rows = []

        for layer_idx, decoder in enumerate(self.model.transformer.dec):
            moe = decoder.mlp

            token_distr = moe.token_distr.detach().cpu().numpy()
            expert_imp = moe.expert_imp.detach().cpu().numpy()
            token_dropped = moe.token_dropped.detach().cpu().numpy()

            if self.model.config.aux_loss_free_load_balance:
                expert_bias = moe.expert_bias.detach().cpu().numpy()
            else:
                expert_bias = [None] * self.model.config.n_experts

            for expert_idx in range(self.model.config.n_experts):
                rows.append(
                    f"{self.step},{layer_idx},{expert_idx},"
                    f"{token_distr[expert_idx]},"
                    f"{expert_imp[expert_idx]},"
                    f"{token_dropped[expert_idx]},"
                    f"{expert_bias[expert_idx]}\n"
                )

        with self.moe_stats_file_path.open("a") as f:
            f.writelines(rows)
        



if __name__=='__main__':

    # from pathlib import Path
    # print(Path.cwd())

    # ======== SET SEED ========
    seed=42
    torch.manual_seed(seed)


    # ======== DEFINE Model Config ========
    llm_config = LLMConfig(
        vocab_size=50257,
        ctx_len=32,
        d_model=64, 
        n_layer=2,
        ff_ratio=4,
        dropout=0.0,
        eps=1e-5,
        bias=False,
        position_embedding='sinusoidal',
        rotary_embedding=False,
        attention='mha',
        normalization='layernorm',
        n_heads=4, 
        n_groups=None,
        use_flash=False, 
        attn_debug=False,
        use_moe=False,
        n_experts=4,
        n_shared_experts=0,
        topk=2,
        capcity_factor = 1.25,
        noisy_router=False,
        router_noise_std=0.0,
        scale_aux_loss_expert_imp=0.0,
        scale_aux_loss_load_balance=0.0,
        aux_loss_free_load_balance=True,
        aux_loss_free_load_balance_bias_update=0.001
    )
    print(llm_config)
    print('-'*50)


    # ======== DEFINE Trainer Config ========
    trainer_config = LLMTrainerConfig(
        num_steps=500,
        batch_size=8,
        learning_rate=6e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        use_lr_scheduler=True,
        warmup_steps=15,
        min_lr=0.1*6e-4,
        grad_clip=1.0,
        log_interval=5,
        moe_log_interval=50,
        eval_interval=50,
        eval_steps=32,
        to_save_checkpoint=False,
        checkpoint_interval=50,
        device='auto'
    )
    print(trainer_config)
    print('-'*50)

    # ======== Instantiate Train Batch Loaders ========
    binary_file_path = './data/tinystories/processed/train.bin'
    tok_bl = TokenBatchLoader(
        B=trainer_config.batch_size, 
        T=llm_config.ctx_len, 
        binary_file_path=binary_file_path, 
        dtype=np.uint16, 
        debug=False
    )

    # ======== Instantiate Val Batch Loaders ========
    binary_file_path = './data/tinystories/processed/val.bin'
    val_token_len = len(np.memmap(filename=binary_file_path, dtype=np.uint16, mode='r'))
    tok_bl_val = TokenBatchLoader(
        B=trainer_config.batch_size, 
        T=llm_config.ctx_len, 
        binary_file_path=binary_file_path, 
        dtype=np.uint16, 
        debug=False
    )

    # ======== Instantiate model ========
    model = LLM(config=llm_config)

    # ======== Instantiate trainer ========
    trainer = LLMTrainer(
        config=trainer_config,
        model=model,
        train_loader=tok_bl,
        val_loader=tok_bl_val
    )


    # ======== Start Training ========
    print('Starting Training...')
    print('-'*50)
    trainer.train()
