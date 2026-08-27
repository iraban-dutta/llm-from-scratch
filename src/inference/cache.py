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
        # During attention, we anyways get the relative position which is always bounded b/w [1, ctx_len-1]
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

            # Update the KV values corresponding to new token 
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




class MHLACacheManager:
    def __init__(self, d_latent:int, d_headR:int, max_new_tokens:int, ctx_len:int, is_rope:bool, n_layer:int):
        # Repeat MHLACache for n_layer
        self.layers = [
            MHLACache(d_latent, d_headR, max_new_tokens, ctx_len, is_rope)
            for _ in range(n_layer)
        ]

    def __getitem__(self, idx:int):
        return self.layers[idx]

    def reset(self):
        for layer in self.layers:
            layer.reset_cache()


class MHLACache:
    def __init__(self, d_latent:int, d_headR:int, max_new_tokens:int, ctx_len:int, is_rope:bool):

        self.d_latent = d_latent
        self.d_headR  = d_headR
        self.max_new_tokens=max_new_tokens
        self.ctx_len=ctx_len
        self.is_rope=is_rope

        # Define tensors to store latent cache and roped_key
        self.latent_kv_cache = None
        self.roped_key_cache = None

        # Current token pos: To track the last token position in the cache
        self.curr_idx = 0
        
        # Total token processed: 
        # If RoPE is enabled, then this is used during inference time to rotate qR, kR vectors by absolute position in sequence
        # This is because we allow longer seqs to occur (by sliding over the latest ctx and storing the latest ctx_len tokens in cache)
        # And we bake the absolute position of the current token in the cache 
        # During attention, we anyways get the relative position which is always bounded b/w [1, ctx_len-1]
        self.ntokens_processed = 0

        

    def _prefill_cache(self, x:torch.Tensor, is_latent:bool):

        if is_latent:
            # latent_kv.shape = (B, T, d_latent)
            B, T, d_l = x.shape

            # Check for latent dim
            assert d_l==(self.d_latent), (
                f"Mismatch in latent dim for KV vectors",
                f"Dim Expected: {self.d_latent}, Dim Got: {d_l}"
            )

            # Define cache vectors of correct shape (prefilled with 0s) 
            T_final = min(self.ctx_len, T+self.max_new_tokens)
            self.latent_kv_cache = torch.zeros(B, T_final, d_l, device=x.device)

            # Update the latent_kv values corresponding to prompt 
            # If prompt has more tokens supported by model, take only the last ctx_len tokens
            T_prompt = min(self.ctx_len, T)
            self.latent_kv_cache[:, :T_prompt] = x[:, -(self.ctx_len):]

        else:
            # kR.shape = (B, 1, T, d_headR)
            B, H, T, d_hR = x.shape

            # Check for kR heads
            assert H == 1, (
                f"Mismatch in KR heads"
                f"#Heads Expected: {1}, #Heads Got: {H}"
            )

            # Check for headR dim
            assert d_hR==(self.d_headR), (
                f"Mismatch in headR dim for roped key (kR)",
                f"Dim Expected: {self.d_headR}, Dim Got: {d_hR}"
            )

            # Define cache vectors of correct shape (prefilled with 0s) 
            T_final = min(self.ctx_len, T+self.max_new_tokens)
            self.roped_key_cache = torch.zeros(B, H, T_final, d_hR, device=x.device)

            # Update the kR values corresponding to prompt 
            # If prompt has more tokens supported by model, take only the last ctx_len tokens
            T_prompt = min(self.ctx_len, T)
            self.roped_key_cache[:, :, :T_prompt] = x[:, :, -(self.ctx_len):]

        # ---------------- XX ----------------
        # Need to be careful with update logic
        # ---------------- XX ----------------
        # When using rope: 
        #     We call update_cache() 2 times in each forward pass (1st: latent_kv and 2nd: roped_key)
        #     We should only update (curr_idx, ntokens_processed) in the 2nd call for each forward pass (when kR gets cached) 
        #     Imagine ntokens_processed gets updated in 1st call -> then we encode wrong positions to kR in 2nd call
        # When NOT using rope:
        #     We call update_cache() 1 time in each forward pass (1st: latent_kv), so we update in 1st call itself
        if self.is_rope and not is_latent:
            # Update pointer for next token position
            self.curr_idx = T_prompt
            # Update ntokens_processed
            self.ntokens_processed += x.shape[2]
        elif not self.is_rope and is_latent:
            # Update pointer for next token position
            self.curr_idx = T_prompt
            # Update ntokens_processed
            self.ntokens_processed += x.shape[2]

        

    def update_cache(self, x:torch.Tensor, is_latent:bool):

        if is_latent and (self.latent_kv_cache) is None:
            # Prefill - Latent KV
            self._prefill_cache(x, is_latent)
        elif not is_latent and (self.roped_key_cache) is None:
            # Prefill - Roped Key
            self._prefill_cache(x, is_latent)
        else:
            # Decode
            if is_latent:
                # latent_kv.shape = (B, T, d_latent)
                B, T, d_l = x.shape      

                # Check for T (should be 1)
                assert T==1, (
                    f"Expected 1 new token to increase cache length by, got {T}"
                )
                
                # Check for latent dim
                assert d_l==(self.d_latent), (
                    f"Mismatch in latent dim for KV vectors",
                    f"Dim Expected: {self.d_latent}, Dim Got: {d_l}"
                )

                # Update the latent_kv values corresponding to new token 
                if self.curr_idx==self.ctx_len:
                    self.latent_kv_cache[:, :-1]=self.latent_kv_cache[:, 1:]
                    self.latent_kv_cache[:, -1:] = x
                else:
                    self.latent_kv_cache[:, self.curr_idx:self.curr_idx+1] = x

            else:
                # kR.shape = (B, 1, T, d_headR)
                B, H, T, d_hR = x.shape

                # Check for T (should be 1)
                assert T==1, (
                    f"Expected 1 new token to increase cache length by, got {T}"
                )

                # Check for kR heads
                assert H == 1, (
                    f"Mismatch in KR heads"
                    f"#Heads Expected: {1}, #Heads Got: {H}"
                )

                # Check for headR dim
                assert d_hR==(self.d_headR), (
                    f"Mismatch in headR dim for roped key (kR)",
                    f"Dim Expected: {self.d_headR}, Dim Got: {d_hR}"
                )

                # Update the kR values corresponding to new token 
                if self.curr_idx==self.ctx_len:
                    self.roped_key_cache[:, :, :-1]=self.roped_key_cache[:, :, 1:]
                    self.roped_key_cache[:, :, -1:] = x
                else:
                    self.roped_key_cache[:, :, self.curr_idx:self.curr_idx+1] = x

            # ---------------- XX ----------------
            # Need to be careful with update logic
            # ---------------- XX ----------------
            # When using rope: 
            #     We call update_cache() 2 times in each forward pass (1st: latent_kv and 2nd: roped_key)
            #     We should only update (curr_idx, ntokens_processed) in the 2nd call for each forward pass (when kR gets cached) 
            #     Imagine ntokens_processed gets updated in 1st call -> then we encode wrong positions to kR in 2nd call
            # When NOT using rope:
            #     We call update_cache() 1 time in each forward pass (1st: latent_kv), so we update in 1st call itself
            if self.is_rope and not is_latent:
                # Update pointer for next token position
                self.curr_idx = min(self.ctx_len, self.curr_idx+1)
                # Update ntokens_processed
                self.ntokens_processed += 1
            elif not self.is_rope and is_latent:
                # Update pointer for next token position
                self.curr_idx = min(self.ctx_len, self.curr_idx+1)
                # Update ntokens_processed
                self.ntokens_processed += 1


    def reset_cache(self):

        # Reset cache to None
        self.latent_kv_cache = None
        self.roped_key_cache = None

        # Reset current token pos
        self.curr_idx = 0
        # Reset number of tokens processed
        self.ntokens_processed = 0




if __name__=='__main__':

    # ================================
    # KV Cache
    # ================================
    # max_new_tokens = 4
    # kv_cache = KVCache(8, 2, max_new_tokens, 12)

    # # Simulate Prefill
    # print('======== Prefill ========')
    # k = torch.randn(3, 2, 10, 8)
    # v = torch.randn(3, 2, 10, 8)

    # # Update KV Cache
    # kv_cache.update_cache(k, v)

    # print(kv_cache.curr_idx)
    # print(kv_cache.k_cache.shape)
    # print(kv_cache.k_cache[0, 0, kv_cache.curr_idx-1])
    # print(kv_cache.k_cache[0, 0, min(kv_cache.ctx_len-1, kv_cache.curr_idx)])


    # # Simulate Decode
    # print('======== Decode ========')
    # for _ in range(max_new_tokens):

    #     k = torch.randn(3, 2, 1, 8)
    #     v = torch.randn(3, 2, 1, 8)

    #     # Update KV Cache
    #     kv_cache.update_cache(k, v)

    #     print(kv_cache.curr_idx)
    #     print(kv_cache.k_cache.shape)
    #     print(kv_cache.k_cache[0, 0, kv_cache.curr_idx-2])
    #     print(kv_cache.k_cache[0, 0, kv_cache.curr_idx-1])
    #     print('-'*50)

    # # Clear Cache
    # kv_cache.reset_cache()
    # print(kv_cache.curr_idx)
    # print(kv_cache.k_cache)


    # ================================
    # MHLA Cache
    # ================================
    max_new_tokens = 4
    mhla_cache = MHLACache(8, 4, max_new_tokens, 12, True)

    # Simulate Prefill
    print('======== Prefill ========')
    latent_kv = torch.randn(3, 10, 8)
    kR        = torch.randn(3, 1, 10, 4)

    # Update Cache - Latent
    mhla_cache.update_cache(latent_kv, is_latent=True)
    print('States post latent update: curr_idx | ntokens_processed\n', mhla_cache.curr_idx, mhla_cache.ntokens_processed)
    if mhla_cache.is_rope:
        # Update Cache - KR
        mhla_cache.update_cache(kR, is_latent=False)
        print('States post rope update: curr_idx | ntokens_processed\n', mhla_cache.curr_idx, mhla_cache.ntokens_processed)
        print('-'*50)

    print(mhla_cache.latent_kv_cache.shape)
    print(mhla_cache.latent_kv_cache[0, mhla_cache.curr_idx-1])
    print(mhla_cache.latent_kv_cache[0, min(mhla_cache.ctx_len-1, mhla_cache.curr_idx)])
    print('-'*50)
    print(mhla_cache.roped_key_cache.shape)
    print(mhla_cache.roped_key_cache[0, 0, mhla_cache.curr_idx-1])
    print(mhla_cache.roped_key_cache[0, 0, min(mhla_cache.ctx_len-1, mhla_cache.curr_idx)])
    print('-'*50)


    # Simulate Decode
    print('======== Decode ========')
    for _ in range(max_new_tokens):

        latent_kv = torch.randn(3, 1, 8)
        kR        = torch.randn(3, 1, 1, 4)

        # Update Cache - Latent
        mhla_cache.update_cache(latent_kv, is_latent=True)
        print('States post latent update: curr_idx | ntokens_processed\n', mhla_cache.curr_idx, mhla_cache.ntokens_processed)
        if mhla_cache.is_rope:
            # Update Cache - KR
            mhla_cache.update_cache(kR, is_latent=False)
            print('States post rope update: curr_idx | ntokens_processed\n', mhla_cache.curr_idx, mhla_cache.ntokens_processed)
            print('-'*50)

        print(mhla_cache.latent_kv_cache.shape)
        print(mhla_cache.latent_kv_cache[0, mhla_cache.curr_idx-2])
        print(mhla_cache.latent_kv_cache[0, mhla_cache.curr_idx-1])
        print('-'*50)
        print(mhla_cache.roped_key_cache.shape)
        print(mhla_cache.roped_key_cache[0, 0, mhla_cache.curr_idx-2])
        print(mhla_cache.roped_key_cache[0, 0, mhla_cache.curr_idx-1])
        print('-'*100)

    # Clear Cache
    mhla_cache.reset_cache()
    print(mhla_cache.curr_idx, mhla_cache.ntokens_processed)
    print(mhla_cache.latent_kv_cache)
    print(mhla_cache.roped_key_cache)