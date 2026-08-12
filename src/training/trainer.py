from pathlib import Path
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
    learning_rate:float = 1e-4
    # weight_decay: float = 0.01
    # beta1: float = 0.9
    # beta2: float = 0.95

    # Logging
    log_interval: int = 10

    # Evaluation
    eval_interval: int = 20
    eval_steps: int    = 16*32 # (16X of train batch size)

    # Checkpointing
    checkpoint_interval: int = 500
    curr_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint_dir: str = f"./checkpoints/{curr_datetime}"

    # Device
    device: str = "auto"

    # Early stopping
    # early_stopping_patience: int | None = None
    
    # # Scheduler
    # warmup_steps: int = ...
    # min_lr: float = ...

    # # Gradient handling
    # grad_clip: float = 1.0





class LLMTrainer:
    def __init__(
        self, 
        config:LLMTrainerConfig, 
        model:nn.Module,
        train_loader:TokenBatchLoader,
        val_loader:TokenBatchLoader
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
        self.optimizer = AdamW(
            self.model.parameters(),
            lr =self.config.learning_rate
        )
        # Track current step of update
        self.step=0

    def _resolve_device(self) -> torch.device:

        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')


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
            "train_loader_state": {
                "curr_idx": self.train_loader.curr_idx,    # Store the curr_idx of train_loader - so that we resume form the same batch
            },
            "rng_state": torch.get_rng_state(),            # While loading: torch.set_rng_state(checkpoint["rng_state"])
        }

        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
        


    def load_checkpoint(self):
        pass


    def model_eval(self) -> float:

        with torch.no_grad():

            # For evaluating over the same #eval_steps batches - Comment our
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


    def train(self):

        val_loss, min_val_loss = 0.0, torch.inf
        # Train Loop
        while self.step < self.config.num_steps:

            # Load Data (x,y) and move tensors to device
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)

            # Zero out gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits, loss = self.model(x, y)

            # Backward Pass
            loss.backward()

            # Optimizer Step
            self.optimizer.step()


            # Logging
            if self.step % self.config.eval_interval == 0:
                val_loss = self.model_eval()
                print(f"Step: {self.step}, Loss: {loss.item():.4f}, Val_Loss: {val_loss:.4f}")
                # Best Val loss obtained
                if val_loss < min_val_loss:
                    self.save_checkpoint(filename='best.pt') # best.pt saved
                    min_val_loss = val_loss
            elif self.step % self.config.log_interval == 0:
                print(f"Step: {self.step}, Loss: {loss.item():.4f}")

            # Checkpointing
            if self.step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(filename='latest.pt') # latest.pt saved

            # Update step attribute    
            self.step += 1

        # Final Logging
        val_loss = self.model_eval()
        print(f"Step: {self.step}, Loss: {loss.item():.4f}, Val_Loss: {val_loss:.4f}")
        # Best Val loss obtained
        if val_loss < min_val_loss:
            self.save_checkpoint(filename='best.pt') # best.pt saved
            min_val_loss = val_loss
        


if __name__=='__main__':

    # from pathlib import Path
    # print(Path.cwd())

    # ======== SET SEED ========
    seed=42
    torch.manual_seed(seed)


    # ======== DEFINE Model Config ========
    llm_config = LLMConfig(
        vocab_size=50257,
        ctx_len=126,
        d_model=384, 
        n_layer=4,
        ff_ratio=4,
        dropout=0.0,
        eps=1e-5,
        position_embedding='sinusoidal',
        rotary_embedding=False,
        attention='mha',
        normalization='layernorm',
        n_heads=4, 
        n_groups=None,
        use_flash=False, 
        attn_debug=False
    )
    print(llm_config)
    print('-'*50)


    # ======== DEFINE Trainer Config ========
    trainer_config = LLMTrainerConfig(
        num_steps=100,
        batch_size=4,
        learning_rate=3e-4,
        log_interval=5,
        eval_interval=40,
        eval_steps=32,
        checkpoint_interval=25,
        device='auto'
    )
    print(llm_config)
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
