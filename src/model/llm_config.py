from dataclasses import dataclass


@dataclass
class LLMConfig:
    # Basic Configs
    vocab_size:int=50257 
    ctx_len:int=8
    d_model:int=32
    n_layer:int=2

    # Components
    position_embedding:str='sinusoidal'
    rotary_embedding:bool=False
    attention:str='mha'
    normalization:str='layernorm'

    # Attention
    n_heads: int = 4
    n_groups: int | None = None


    def __post_init__(self):

        # Run validation checks on the config passed
        self.validate()


    def validate(self) -> None:

        # ======== VALIDATE BASIC CONFIGS ========
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.ctx_len <= 0:
            raise ValueError("ctx_len must be > 0.")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0.")
        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even.")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be > 0.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0.")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )

        # ======== VALIDATE COMPONENTS ========
        # Validate position_embedding
        valid_pos_enc = {
            "sinusoidal",
            "learned",
            "identity"
        }
        if self.position_embedding not in valid_pos_enc:
            raise ValueError(
                f"Unknown position_embedding '{self.position_embedding}'. "
                f"Expected one of {valid_pos_enc}"
            )

        # If rotary_embedding is True , overrride position_embedding to identity
        if self.rotary_embedding:
            self.position_embedding = "identity"

        # Validate attention
        valid_attention = {
            "mha",
            "mqa",
            "gqa",
            "mhla",
        }
        if self.attention not in valid_attention:
            raise ValueError(
                f"Unknown attention '{self.attention}'. "
                f"Expected one of {valid_attention}"
            )
        
        # Validate normalization
        valid_norm = {
            "layernorm",
            "rmsnorm",
        }
        if self.normalization not in valid_norm:
            raise ValueError(
                f"Unknown normalization '{self.normalization}'. "
                f"Expected one of {valid_norm}"
            )

        # ======== VALIDATE ATTENTION ========
        match self.attention:

            case "mha":
                pass
            case "mqa":
                pass
            case "gqa":
                if self.n_groups is None:
                    raise ValueError(
                        "GQA requires n_groups."
                    )
                if self.n_groups <= 0:
                    raise ValueError(
                        "n_groups must be > 0."
                    )
                if self.n_heads % self.n_groups != 0:
                    raise ValueError(
                        "n_heads must be divisible by n_groups."
                    )
            case "mhla":
                pass