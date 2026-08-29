from __future__ import annotations

import torch

from apex.inference import generate
from apex.model import ApexConfig, ApexForCausalLM
from apex.training import Trainer, TrainerConfig


def test_transformer_forward_loss_weight_tying_and_cache() -> None:
    torch.manual_seed(7)
    config = ApexConfig.tiny(vocab_size=300)
    model = ApexForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    output = model(input_ids, labels=input_ids, use_cache=True)
    assert output.logits.shape == (2, 12, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.past_key_values is not None
    assert len(output.past_key_values) == config.num_layers
    assert model.lm_head.weight.data_ptr() == model.token_embeddings.weight.data_ptr()

    next_token = torch.randint(0, config.vocab_size, (2, 1))
    cached = model(next_token, past_key_values=output.past_key_values, use_cache=True)
    assert cached.logits.shape == (2, 1, config.vocab_size)
    assert cached.past_key_values is not None
    assert cached.past_key_values[0][0].shape[2] == 13


def test_training_checkpoint_and_seeded_generation(tmp_path) -> None:
    config = ApexConfig.tiny(vocab_size=300)
    model = ApexForCausalLM(config)
    trainer = Trainer(model, TrainerConfig(learning_rate=1e-3))
    batch = torch.randint(0, config.vocab_size, (1, 16))
    loss = trainer.train_step(batch)
    assert loss > 0
    checkpoint = tmp_path / "tiny.pt"
    trainer.save_checkpoint(checkpoint, metadata={"purpose": "test"})
    assert trainer.load_checkpoint(checkpoint) == {"purpose": "test"}

    model.eval()
    first = generate(model, batch[:, :3], max_new_tokens=3, top_k=8, seed=11)
    second = generate(model, batch[:, :3], max_new_tokens=3, top_k=8, seed=11)
    assert first.shape == (1, 6)
    assert torch.equal(first, second)
