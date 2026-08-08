import torch.nn as nn
import torch.nn.functional as F
from .llm_config import LLMConfig
from .position_embedding import build_position_embedding


class LLM(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.config=config

        model_dict = {
            "wte":nn.Embedding(config.vocab_size, config.d_model),
            "wpe":build_position_embedding(config),

        }
        
        self.transformer = nn.ModuleDict(model_dict)


if __name__=='__main__':

    config = LLMConfig(
        position_embedding='sinusoidal', 
        rotary_embedding=False
    )
    print(config)

    model = LLM(config)
    print(model.transformer)
    for k,v in model.state_dict().items():
        print(k, v.shape)

