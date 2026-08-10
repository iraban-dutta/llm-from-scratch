from typing import Any


def tokenize_tinystories(story:str, tokenizer:Any) -> list[int]:
    if (
        not hasattr(tokenizer, "eot_token")
        or tokenizer.eot_token != 50256
    ):
        raise ValueError(
            "Expected a tiktoken GPT-2 Encoding object "
            "(eot_token should be 50256)."
        )

    story_tokens = tokenizer.encode(story)
    # story_tokens.append(tokenizer.eot_token)
    return story_tokens
