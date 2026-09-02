import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from .llm_config import LLMConfig
from typing import Tuple, List


class MLP(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.proj_in=nn.Linear(in_features=config.d_model, out_features=config.ff_ratio*config.d_model, bias=config.bias)
        self.gelu=nn.GELU(approximate='tanh')
        self.proj_out=nn.Linear(in_features=config.ff_ratio*config.d_model, out_features=config.d_model, bias=config.bias)
        self.proj_out.RESIDUAL_PATH_SCALE_INIT=1

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.gelu(x)
        x = self.proj_out(x)
        return x


class MoE(nn.Module):

    def __init__(self, config:LLMConfig):
        super().__init__()

        # MoE
        self.n_experts=config.n_experts
        self.experts = nn.ModuleList(
            [MLP(config) for _ in range(config.n_experts)]
        )
        self.capacity_factor=config.capacity_factor

        # Shared experts
        self.n_shared_experts=config.n_shared_experts
        if self.n_shared_experts>0:
            self.shared_experts = nn.ModuleList(
                [MLP(config) for _ in range(config.n_shared_experts)]
            )
        else:
            self.shared_experts = None

        # Router
        self.router = nn.Linear(config.d_model, config.n_experts)
        self.router_noise_std = config.router_noise_std if config.noisy_router else 0.0
        self.topk = config.topk

        # Expert level Diagnostics
        self.register_buffer('expert_imp', torch.zeros(config.n_experts))
        self.register_buffer('token_distr', torch.zeros(config.n_experts))
        self.register_buffer('load_balance', torch.zeros(config.n_experts))
        self.register_buffer('expert_bias', torch.zeros(config.n_experts))
        self.register_buffer('token_dropped', torch.zeros(config.n_experts))

        # Balanced MoE
        self.scale_aux_loss_cv=config.scale_aux_loss_expert_imp
        self.scale_aux_loss_lb=config.scale_aux_loss_load_balance
        self.loss_coeff_var = 0.0
        self.loss_load_balance = 0.0
        self.aux_loss_free_lb=config.aux_loss_free_load_balance
        self.aux_loss_free_lb_bias_updt=config.aux_loss_free_load_balance_bias_update

        # Populated only by forward_benchmark(); unused during normal training.
        self.last_forward_bench_ms = {}

        # Flip to False for legacy dispatch benchmark
        self.use_vectorized_dispatch = config.use_vectorized_dispatch  


    def get_topk_from_router(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # ----------------------------------------------------------------
        # Get router logits | shape: (nT, nE)
        # ----------------------------------------------------------------
        router_logits = self.router(x)

        # ----------------------------------------------------------------
        # Add noise to router logits (if noisy_router=False, noise added is 0)
        # ----------------------------------------------------------------
        router_logits = router_logits + (self.router_noise_std * torch.randn_like(router_logits))

        # ----------------------------------------------------------------
        # Add dynamic bias correction: Achieves aux loss free load balance 
        # (if aux_loss_free_lb=False or model in eval mode, bias added is 0)
        # ----------------------------------------------------------------
        router_logits = router_logits + self.expert_bias

        # ----------------------------------------------------------------
        # Get topk vals and idxs | shape: (nT, topk)
        # ----------------------------------------------------------------
        topk_vals, topk_idxs = torch.topk(router_logits, k=self.topk, dim=-1)

        # ----------------------------------------------------------------
        # Normalize topk_vals | shape: (nT, topk)
        # ----------------------------------------------------------------
        topk_vals = F.softmax(topk_vals, dim=-1)

        return topk_vals, topk_idxs


    def _update_expert_diagnostics_vectorized(
        self,
        router_ohe: torch.Tensor,         # (topk, T, nE) int
        router_ohe_vals: torch.Tensor,    # (topk, T, nE) float
        T: int,
        capacity: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All-experts diagnostics before capacity clip - For vectorized dispatch"""

        slot_counts = router_ohe.sum(dim=(0, 1))              # (nE,)
        expert_imp = router_ohe_vals.sum(dim=(0, 1))          # (nE,), keeps grad
        token_distr = slot_counts.to(expert_imp.dtype) / (T * self.topk)
        with torch.no_grad():
            self.token_dropped.copy_(torch.clamp(slot_counts.float() - capacity, min=0))
            self.expert_imp.copy_(expert_imp.detach())
            self.token_distr.copy_(token_distr)
            self.load_balance.copy_((expert_imp.detach() / T) * token_distr)
            if self.aux_loss_free_lb and self.training:
                target = (T * self.topk) / self.n_experts
                signs = torch.sign(target - slot_counts.float())
                self.expert_bias.add_(self.aux_loss_free_lb_bias_updt * signs)
        return expert_imp, token_distr


    def _dispatch_tokens_vectorized(self, topk_idxs:torch.Tensor, topk_vals:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ----------------------------------------------------------------
        # Compute capacity: C = (nT*topk/nE)*capacity_factor
        # ----------------------------------------------------------------
        T = topk_idxs.shape[0] # Total Tokens
        capacity = math.ceil(((T*self.topk)/self.n_experts)*self.capacity_factor)

        # ----------------------------------------------------------------
        # topk_idxs_reshaped: (topk, T, 1)
        # topk_vals_reshaped: (topk, T, 1)
        # ----------------------------------------------------------------
        topk_idxs_reshaped = torch.stack(torch.split(topk_idxs, 1, dim=-1), dim=0)
        topk_vals_reshaped = torch.stack(torch.split(topk_vals, 1, dim=-1), dim=0)

        # ----------------------------------------------------------------
        # OHE Tensor creation
        # ----------------------------------------------------------------
        device = topk_idxs.device
        dtype = topk_vals.dtype

        # Scatter 1s to the relevant expert idxs for each token
        # shape: (topk, T, n_experts)
        router_ohe = torch.zeros(self.topk, T, self.n_experts, dtype=torch.int32, device=device)
        router_ohe.scatter_(-1, topk_idxs_reshaped, torch.ones_like(topk_idxs_reshaped, dtype=torch.int32))

        # Scatter the expert weights to the relevant expert idxs for each token
        # shape: (topk, T, n_experts)
        router_ohe_vals = torch.zeros(self.topk, T, self.n_experts, dtype=dtype, device=device)
        router_ohe_vals.scatter_(-1, topk_idxs_reshaped, topk_vals_reshaped)

        # ----------------------------------------------------------------
        # Update expert level diagnostics HERE, before capacity
        # ----------------------------------------------------------------
        expert_imp, token_distr = self._update_expert_diagnostics_vectorized(
            router_ohe, router_ohe_vals, T, capacity
        )

        # ----------------------------------------------------------------
        # Adjust for capacity: 
        # Maintain priority in which tokens are routed to experts
        # shape: (T*topk, n_experts)
        # ----------------------------------------------------------------
        router_ohe_flattened = router_ohe.view(-1, self.n_experts)
        router_ohe_flattened = router_ohe_flattened.cumsum(dim=0)-1
        capacity_mask = router_ohe_flattened<capacity

        # ----------------------------------------------------------------
        # Adjust the router_ohe_vals for capacity
        # shape: (T*topk, n_experts)
        # ----------------------------------------------------------------
        router_ohe_vals_flattened = router_ohe_vals.view(-1, self.n_experts)
        router_ohe_vals_flattened_cap_adjusted = router_ohe_vals_flattened * capacity_mask

        # ----------------------------------------------------------------
        # Get final router 
        # shape: (T*topk, n_experts) -> (topk, T, n_experts) -> (T, n_experts)
        # ----------------------------------------------------------------
        router_final = router_ohe_vals_flattened_cap_adjusted.view(self.topk, -1, self.n_experts)
        router_final = router_final.sum(dim=0)

        return router_final, expert_imp, token_distr

    
    def _dispatch_tokens(self, expert_idx:int, topk_idxs:torch.Tensor, topk_vals:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

        # ----------------------------------------------------------------
        # Compute capacity: C = (nT*topk/nE)*capacity_factor
        # ----------------------------------------------------------------
        T = topk_idxs.shape[0] # Total Tokens
        capacity = math.ceil(((T*self.topk)/self.n_experts)*self.capacity_factor)

        token_idxs = []
        token_vals = []

        # ----------------------------------------------------------------
        # Retrieve the token_idxs & their corresponding vals routed to expert_i
        # ----------------------------------------------------------------
        # idx_dim0 can be token position
        # idx_dim1 can be anything b/w [0, k-1]
        idx_dim0, idx_dim1 = torch.where(topk_idxs==expert_idx)
        for k in range(self.topk):
            # Retrieve the token_idxs & their corresponding vals routed to expert_i @ pos=k
            token_idxs_ik = idx_dim0[idx_dim1==k]
            token_vals_ik = topk_vals[token_idxs_ik, k]
            token_idxs.append(token_idxs_ik)
            token_vals.append(token_vals_ik)

        token_idxs = torch.cat(token_idxs, dim=0)
        token_vals = torch.cat(token_vals, dim=0)

        # ----------------------------------------------------------------
        # Update Expert level Diagnostics (before capacity clip) 
        # ----------------------------------------------------------------
        expert_imp = token_vals.sum()
        token_distr = token_idxs.shape[0] / (T * self.topk)
        with torch.no_grad():
            # token_dropped[i]: Number of tokens routed to this expert but will be dropped due to capacity.
            self.token_dropped[expert_idx] = max(0, token_idxs.shape[0] - capacity)
            # expert_importance[i]
            self.expert_imp[expert_idx] = expert_imp.detach()
            # token_distribution[i]
            self.token_distr[expert_idx] = token_distr
            # load_balance[i]
            self.load_balance[expert_idx] = (expert_imp.detach() / T) * token_distr
            # Aux Loss Free Load Balance
            if self.aux_loss_free_lb and self.training:
                expert_overload_sign = np.sign( (T*self.topk) / (self.n_experts) - token_idxs.shape[0])
                self.expert_bias[expert_idx] += self.aux_loss_free_lb_bias_updt * expert_overload_sign

        return (token_idxs[:capacity], token_vals[:capacity], expert_imp, token_distr)


    def _run_experts(self, expert_idx:int, x:torch.Tensor, expert_token_idxs:torch.Tensor) -> torch.Tensor:

        # ----------------------------------------------------------------
        # Puck out token_idxs from master x
        # ----------------------------------------------------------------
        expert_inp = x[expert_token_idxs]

        # ----------------------------------------------------------------
        # Forward Pass via expert_i
        # ----------------------------------------------------------------
        expert_out = self.experts[expert_idx](expert_inp)

        return expert_out
        

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        B, S, D = x.shape

        # ----------------------------------------------------------------
        # Flatten
        # ----------------------------------------------------------------
        # shape: (nT, D) where nT=B*S
        x = x.view(-1, D)

        # ----------------------------------------------------------------
        # Router-TopK 
        # ----------------------------------------------------------------
        # shape: (nT, topk)
        topk_vals, topk_idxs = self.get_topk_from_router(x)

        # ----------------------------------------------------------------
        # Define output tensor
        # ----------------------------------------------------------------
        # shape: (nT, D)
        out = torch.zeros_like(x)

        # ----------------------------------------------------------------
        # Shared experts: 
        # ----------------------------------------------------------------
        if self.n_shared_experts>0:
            for expert_idx in range(self.n_shared_experts):
                out += self.shared_experts[expert_idx](x)


        if self.use_vectorized_dispatch:
            # ================================================================
            # Forward Pass via vectorized dispatch
            # ================================================================

            # ----------------------------------------------------------------
            # Vectorized dispatch
            # ----------------------------------------------------------------
            router_final, expert_imp, token_distr = self._dispatch_tokens_vectorized(
                topk_idxs, topk_vals
            )

            # ----------------------------------------------------------------
            # Expert wise Forward Pass
            # ----------------------------------------------------------------
            for expert_idx in range(self.n_experts):
                expert_wts = router_final[:, expert_idx]   # shape (T,)
                routed = expert_wts != 0
                expert_token_idxs = torch.arange(expert_wts.size(0), device=expert_wts.device)[routed]
                expert_token_vals = expert_wts[routed]   
                if expert_token_idxs.numel() > 0:
                    expert_out = self._run_experts(expert_idx, x, expert_token_idxs)
                    expert_out = expert_out * expert_token_vals.unsqueeze(-1)
                    out[expert_token_idxs] += expert_out

            # ----------------------------------------------------------------
            # Losses for balanced MoE
            # ----------------------------------------------------------------
            # aux loss — already have full vectors, no per-expert stack
            load_balance = (expert_imp / x.shape[0]) * token_distr
            self.loss_coeff_var = self.scale_aux_loss_cv * (
                expert_imp.std() / (expert_imp.mean() + 1e-5)
            )
            self.loss_load_balance = self.scale_aux_loss_lb * self.n_experts * load_balance.sum()

        else:
            # ================================================================
            # Forward Pass via legacy dispatch
            # ================================================================

            # ----------------------------------------------------------------
            # Loop over each expert: 
            # ----------------------------------------------------------------
            # 1: Run dispatcher: Get tokens routed to expert_i in correct order and upto capacity
            # 2: Run Forward and scale as per (expert_id x token_id) weight
            # 3: Collate expert output
            expert_imp = []
            token_distr = []
            for expert_idx in range(self.n_experts):

                # Dispatcher - get token_ids routed to expert_i in correct order and upto capacity
                # expert_token_idxs.shape: (nt,)
                # expert_token_vals.shape: (nt,)
                # nt is bounded within [0, capacity]
                expert_token_idxs, expert_token_vals, expert_imp_i, token_distr_i = self._dispatch_tokens(expert_idx, topk_idxs, topk_vals)

                expert_imp.append(expert_imp_i)
                token_distr.append(token_distr_i)

                # Run only if current expert has tokens dispatched to it
                if len(expert_token_idxs)>0:
                    
                    # Forward
                    # shape: (nt, D)
                    expert_out = self._run_experts(expert_idx, x, expert_token_idxs)

                    # Scale expert_out
                    # shape: (nt, D)
                    expert_out = expert_out * expert_token_vals.unsqueeze(-1)

                    # Combine tokens
                    # shape: (nt, D)
                    out[expert_token_idxs] += expert_out

            # ----------------------------------------------------------------
            # Losses for balanced MoE
            # ----------------------------------------------------------------
            # Stack scalar tensors into one tensor while preserving their autograd graph.
            expert_imp = torch.stack(expert_imp)

            # Convert routing fractions into a tensor; these are non-differentiable statistics.
            token_distr = torch.tensor(token_distr, device=x.device, dtype=expert_imp.dtype,)

            # Compute per-expert load-balance term from expert importance and token fraction.
            load_balance = (expert_imp/x.shape[0]) * token_distr

            # Coefficient-of-variation auxiliary loss; remains connected to the routing graph.
            self.loss_coeff_var = self.scale_aux_loss_cv * (expert_imp.std()/(expert_imp.mean() + 1e-5))

            # Load-balance auxiliary loss; remains connected to the routing graph.
            self.loss_load_balance = self.scale_aux_loss_lb * self.n_experts * load_balance.sum()

        return out.view(B, S, D)


    @staticmethod
    def _sync_tensor(t: torch.Tensor) -> None:
        if t.is_cuda:
            torch.cuda.synchronize()
        elif t.device.type == 'mps':
            torch.mps.synchronize()
    def forward_benchmark(self, x: torch.Tensor) -> torch.Tensor:
        """Same compute graph as forward(), with per-phase timing for benchmarking."""
        bench = {
            'route_ms': 0.0,
            'dispatch_ms': 0.0,
            'expert_ms': 0.0,
            'combine_ms': 0.0,
            'shared_ms': 0.0,
            'aux_ms': 0.0,
        }
        B, S, D = x.shape
        self._sync_tensor(x)
        t0 = time.perf_counter()
        x = x.view(-1, D)
        topk_vals, topk_idxs = self.get_topk_from_router(x)
        self._sync_tensor(x)
        bench['route_ms'] = (time.perf_counter() - t0) * 1000
        out = torch.zeros_like(x)
        if self.n_shared_experts > 0:
            self._sync_tensor(x)
            t0 = time.perf_counter()
            for expert_idx in range(self.n_shared_experts):
                out += self.shared_experts[expert_idx](x)
            self._sync_tensor(x)
            bench['shared_ms'] = (time.perf_counter() - t0) * 1000
        if self.use_vectorized_dispatch:
            # ----------------------------------------------------------------
            # Vectorized dispatch + per-expert token gather
            # ----------------------------------------------------------------
            self._sync_tensor(x)
            t0 = time.perf_counter()
            router_final, expert_imp, token_distr = self._dispatch_tokens_vectorized(
                topk_idxs, topk_vals
            )
            self._sync_tensor(x)
            bench['dispatch_ms'] = (time.perf_counter() - t0) * 1000
            for expert_idx in range(self.n_experts):
                # Gather routed tokens (legacy counts this inside _dispatch_tokens)
                self._sync_tensor(x)
                t0 = time.perf_counter()
                expert_wts = router_final[:, expert_idx]
                routed = expert_wts != 0
                expert_token_idxs = torch.arange(
                    expert_wts.size(0), device=expert_wts.device
                )[routed]
                expert_token_vals = expert_wts[routed]
                self._sync_tensor(x)
                bench['dispatch_ms'] += (time.perf_counter() - t0) * 1000
                if expert_token_idxs.numel() > 0:
                    self._sync_tensor(x)
                    t0 = time.perf_counter()
                    expert_out = self._run_experts(expert_idx, x, expert_token_idxs)
                    self._sync_tensor(x)
                    bench['expert_ms'] += (time.perf_counter() - t0) * 1000
                    self._sync_tensor(x)
                    t0 = time.perf_counter()
                    expert_out = expert_out * expert_token_vals.unsqueeze(-1)
                    out[expert_token_idxs] += expert_out
                    self._sync_tensor(x)
                    bench['combine_ms'] += (time.perf_counter() - t0) * 1000
            self._sync_tensor(x)
            t0 = time.perf_counter()
            load_balance = (expert_imp / x.shape[0]) * token_distr
            self.loss_coeff_var = self.scale_aux_loss_cv * (
                expert_imp.std() / (expert_imp.mean() + 1e-5)
            )
            self.loss_load_balance = self.scale_aux_loss_lb * self.n_experts * load_balance.sum()
            self._sync_tensor(x)
            bench['aux_ms'] = (time.perf_counter() - t0) * 1000
        else:
            # ----------------------------------------------------------------
            # Legacy dispatch
            # ----------------------------------------------------------------
            expert_imp = []
            token_distr = []
            for expert_idx in range(self.n_experts):
                self._sync_tensor(x)
                t0 = time.perf_counter()
                expert_token_idxs, expert_token_vals, expert_imp_i, token_distr_i = (
                    self._dispatch_tokens(expert_idx, topk_idxs, topk_vals)
                )
                self._sync_tensor(x)
                bench['dispatch_ms'] += (time.perf_counter() - t0) * 1000
                expert_imp.append(expert_imp_i)
                token_distr.append(token_distr_i)
                if expert_token_idxs.numel() > 0:
                    self._sync_tensor(x)
                    t0 = time.perf_counter()
                    expert_out = self._run_experts(expert_idx, x, expert_token_idxs)
                    self._sync_tensor(x)
                    bench['expert_ms'] += (time.perf_counter() - t0) * 1000
                    self._sync_tensor(x)
                    t0 = time.perf_counter()
                    expert_out = expert_out * expert_token_vals.unsqueeze(-1)
                    out[expert_token_idxs] += expert_out
                    self._sync_tensor(x)
                    bench['combine_ms'] += (time.perf_counter() - t0) * 1000
            self._sync_tensor(x)
            t0 = time.perf_counter()
            expert_imp = torch.stack(expert_imp)
            token_distr = torch.tensor(token_distr, device=x.device, dtype=expert_imp.dtype)
            load_balance = (expert_imp / x.shape[0]) * token_distr
            self.loss_coeff_var = self.scale_aux_loss_cv * (
                expert_imp.std() / (expert_imp.mean() + 1e-5)
            )
            self.loss_load_balance = self.scale_aux_loss_lb * self.n_experts * load_balance.sum()
            self._sync_tensor(x)
            bench['aux_ms'] = (time.perf_counter() - t0) * 1000
        self.last_forward_bench_ms = bench
        return out.view(B, S, D)


if __name__=='__main__':

    # MoE definition
    config = LLMConfig(
        use_moe=True,
        n_experts=3,
        n_shared_experts=1,
        topk=2,
        d_model=4,
        ff_ratio=4,
        capacity_factor = 1.2,
        noisy_router=False,
        router_noise_std=0.1,
        scale_aux_loss_expert_imp=1.0,
        scale_aux_loss_load_balance=1.0,
        aux_loss_free_load_balance=False,
        aux_loss_free_load_balance_bias_update=0.01
    )

    # ----------------------------------------------------------------
    # Test Forward Pass
    # ----------------------------------------------------------------
    # moe = MoE(config)

    # # Forward Pass
    # g = torch.Generator().manual_seed(42)
    # x_inp = torch.randn(2, 3, 4, generator=g)
    # print(x_inp.shape)
    # print(x_inp)
    # print("="*25, 'Start Forward', "="*25)
    # moe_xout = moe(x_inp)
    # print("="*25, 'End Forward', "="*25)
    # print(moe_xout.shape)
    # print(moe_xout)
    # print('-'*50)

    # print('moe.token_dropped:\n', moe.token_dropped)
    # print('moe.expert_imp:\n', moe.expert_imp)
    # print('moe.token_distr:\n', moe.token_distr)
    # print('moe.load_balance:\n', moe.load_balance)
    # print('moe.expert_bias:\n', moe.expert_bias)

    # print('moe.loss_coeff_var:\n', moe.loss_coeff_var.item())
    # print('moe.loss_load_balance:\n', moe.loss_load_balance.item())

    # ----------------------------------------------------------------
    # Test vectorized dispatch vs legacy dispatch
    # ----------------------------------------------------------------
    moe_v = MoE(config); moe_l = MoE(config)
    moe_l.load_state_dict(moe_v.state_dict())
    moe_v.use_vectorized_dispatch = True
    moe_l.use_vectorized_dispatch = False

    g = torch.Generator().manual_seed(42)
    x = torch.randn(2, 3, 4, generator=g)
    print(x.shape)
    print(x)

    print("="*25, 'Start Forward', "="*25)
    out_v = moe_v(x)
    out_l = moe_l(x)
    print("="*25, 'End Forward', "="*25)
    print(out_v.shape)
    print(out_v)
    print(out_l.shape)
    print(out_l)
    print('-'*50)
    assert torch.allclose(out_v, out_l, atol=1e-5)
    assert torch.allclose(moe_v.expert_imp, moe_l.expert_imp, atol=1e-5)