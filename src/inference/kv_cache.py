import torch


class KVCacheManager:
    def __init__(self, head_dim:int, n_kv:int, max_new_tokens:int, ctx_len:int, n_layer:int):
        # Repeat KVCache for n_layer
        self.layers = [
            KVCache(head_dim, n_kv, max_new_tokens, ctx_len)
            for _ in range(n_layer)
        ]

    def __getitem__(self, idx:int):
        return self.layers[idx]

    def reset(self):
        for layer in self.layers:
            layer.reset_cache()


class KVCache:
    def __init__(self, head_dim:int, n_kv:int, max_new_tokens:int, ctx_len:int):

        self.n_kv = n_kv
        self.head_dim = head_dim
        self.max_new_tokens=max_new_tokens
        self.ctx_len=ctx_len

        # Define tensors to store k and v vectors of past tokens
        self.k_cache = None
        self.v_cache = None

        # Current token pos: To track the last token position in the cache
        self.curr_idx = 0
        
        # Total token processed: 
        # If RoPE is enabled, then this is used during inference time to rotate q,k vectors by absolute position in sequence
        # This is because we allow longer seqs to occur (by sliding over the latest ctx and storing the latest ctx_len tokens in cache)
        # And we bake the absolute position of the current token in the cache 
        # During attention, we anyways get the relative position which is alwats bounded b/w [1, ctx_len-1]
        self.ntokens_processed = 0

        

    def _prefill_cache(self, k:torch.Tensor, v:torch.Tensor):

        # k.shape = (B, n_kv, T, head_dim)
        # v.shape = (B, n_kv, T, head_dim)
        B, n_kv, T, d_kv = k.shape

        # Check for kv heads
        assert self.n_kv == n_kv, (
            f"Mismatch in KV heads"
            f"#Heads Expected: {self.n_kv}, #Heads Got: {n_kv}"
        )

        # Check for kv dims
        assert d_kv==(self.head_dim), (
            f"Mismatch in head dim for KV vectors",
            f"Dim Expected: {self.head_dim}, Dim Got: {d_kv}"
        )

        # Define cache vectors of correct shape (prefilled with 0s) 
        T_final = min(self.ctx_len, T+self.max_new_tokens)
        self.k_cache = torch.zeros(B, n_kv, T_final, d_kv, device=k.device)
        self.v_cache = torch.zeros(B, n_kv, T_final, d_kv, device=v.device)

        # Update the KV values corresponding to prompt 
        # If prompt has more tokens supported by model, take only the last ctx_len tokens
        T_prompt = min(self.ctx_len, T)
        self.k_cache[:, :, :T_prompt] = k[:, :, -(self.ctx_len):]
        self.v_cache[:, :, :T_prompt] = v[:, :, -(self.ctx_len):]

        # Update pointer for next token position
        self.curr_idx = T_prompt

        # Update ntokens_processed
        self.ntokens_processed += k.shape[2]

    def update_cache(self, k:torch.Tensor, v:torch.Tensor):

        if self.k_cache is None:
            # Prefill
            self._prefill_cache(k, v)
        else:
            # Decode
            # k.shape = (B, n_kv, 1, n_kv*head_dim)
            # v.shape = (B, n_kv, 1, n_kv*head_dim)
            B, n_kv, T, d_kv = k.shape

            # Check for kv heads
            assert self.n_kv == n_kv, (
                f"Mismatch in KV heads"
                f"#Heads Expected: {self.n_kv}, #Heads Got: {n_kv}"
            )

            # Check for kv dims
            assert d_kv==(self.head_dim), (
                f"Mismatch in hidden dim for KV vectors (concated across all heads)",
                f"Dim Expected: {self.head_dim}, Dim Got: {d_kv}"
            )

            # Check for T (should be 1)
            assert T==1, (
                f"Expected 1 new token to increase KV cache length by, got {T}"
            )

            # Update the KV values corresponding to new 
            if self.curr_idx==self.ctx_len:
                self.k_cache[:, :, :-1]=self.k_cache[:, :, 1:]
                self.v_cache[:, :, :-1]=self.v_cache[:, :, 1:]
                self.k_cache[:, :, -1:] = k
                self.v_cache[:, :, -1:] = v
            else:
                self.k_cache[:, :, self.curr_idx:self.curr_idx+1] = k
                self.v_cache[:, :, self.curr_idx:self.curr_idx+1] = v

            # Update pointer for next token position
            self.curr_idx = min(self.ctx_len, self.curr_idx+1)

            # Update ntokens_processed
            self.ntokens_processed += 1


    def reset_cache(self):

        # Reset KV cache to None
        self.k_cache = None
        self.v_cache = None

        # Reset current token pos
        self.curr_idx = 0
        # Reset number of tokens processed
        self.ntokens_processed = 0



if __name__=='__main__':

    max_new_tokens = 4
    kv_cache = KVCache(8, 2, max_new_tokens, 12)

    # Simulate Prefill
    print('======== Prefill ========')
    k = torch.randn(3, 2, 10, 8)
    v = torch.randn(3, 2, 10, 8)

    # Update KV Cache
    kv_cache.update_cache(k, v)

    print(kv_cache.curr_idx)
    print(kv_cache.k_cache.shape)
    print(kv_cache.k_cache[0, 0, kv_cache.curr_idx-1])
    print(kv_cache.k_cache[0, 0, min(kv_cache.ctx_len-1, kv_cache.curr_idx)])


    # Simulate Decode
    print('======== Decode ========')
    for _ in range(max_new_tokens):

        k = torch.randn(3, 2, 1, 8)
        v = torch.randn(3, 2, 1, 8)

        # Update KV Cache
        kv_cache.update_cache(k, v)

        print(kv_cache.curr_idx)
        print(kv_cache.k_cache.shape)
        print(kv_cache.k_cache[0, 0, kv_cache.curr_idx-2])
        print(kv_cache.k_cache[0, 0, kv_cache.curr_idx-1])
        print('-'*50)

    # Clear Cache
    kv_cache.reset_cache()
    print(kv_cache.curr_idx)
    print(kv_cache.k_cache)