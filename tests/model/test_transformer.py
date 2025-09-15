import torch

from amlm.model.config import DualFFNConfig, ModelConfig
from amlm.model.transformer import TransformerModel


def sample_config() -> ModelConfig:
    return ModelConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_size=64,
        vocab_size=100,
        max_position_embeddings=128,
        dual_ffn=DualFFNConfig(enabled=True, small_ratio=0.5),
    )


def test_transformer_forward_shape() -> None:
    cfg = sample_config()
    model = TransformerModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    logits, caches, stats = model(input_ids)
    assert logits.shape == (2, 5, cfg.vocab_size)
    assert len(caches) == cfg.n_layers
    assert len(stats) == cfg.n_layers


def test_transformer_cache_progression() -> None:
    cfg = sample_config()
    model = TransformerModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 1))
    logits, caches, _ = model(input_ids, use_cache=True)
    assert caches[0] is not None
    next_ids = torch.randint(0, cfg.vocab_size, (1, 1))
    logits2, caches2, _ = model(next_ids, past_key_values=caches, use_cache=True)
    assert caches2[0][0].shape[-2] == 2
    assert logits2.shape == (1, 1, cfg.vocab_size)
