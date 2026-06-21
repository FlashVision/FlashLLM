"""Unit tests for generation components."""

import torch

from flashllm.generation.sampler import Sampler


def test_sampler_greedy():
    sampler = Sampler(temperature=1.0, top_k=0, top_p=1.0)
    logits = torch.tensor([[-1.0, 0.5, 2.0, -0.5, 1.0]])
    token = sampler.greedy(logits)
    assert token.item() == 2


def test_sampler_top_k():
    sampler = Sampler(temperature=1.0, top_k=2, top_p=1.0)
    logits = torch.tensor([[1.0, 5.0, 3.0, 0.1, 0.2]])
    token = sampler(logits)
    assert token.item() in [1, 2]


def test_sampler_temperature():
    sampler_cold = Sampler(temperature=0.01, top_k=0, top_p=1.0)
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.3, 0.1]])
    token = sampler_cold(logits)
    assert token.item() == 2


def test_sampler_repetition_penalty():
    sampler = Sampler(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=100.0)
    logits = torch.tensor([[5.0, 0.1, 0.1, 0.1, 0.1]])
    generated = torch.tensor([0])
    token = sampler(logits, generated_ids=generated)
    assert token.item() != 0


def test_sampler_output_shape():
    sampler = Sampler()
    logits = torch.randn(1, 32000)
    token = sampler(logits)
    assert token.shape == (1,)


def test_sampler_batch():
    sampler = Sampler(temperature=0.5, top_k=10)
    logits = torch.randn(4, 32000)
    tokens = sampler(logits)
    assert tokens.shape == (4,)
