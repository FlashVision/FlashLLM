"""Tests for new FlashLLM P0 components — MoE, RoPE scaling, GaLore, quantization, eval, etc."""

import pytest
import torch

from flashllm.models.moe import TopKRouter, Expert, MoELayer, MoETransformerBlock
from flashllm.models.rope_scaling import (
    LinearScaledRoPE, DynamicNTKScaledRoPE, YaRNScaledRoPE, get_rope_scaling,
)
from flashllm.generation.function_calling import (
    FunctionSchema, FunctionParameter, FunctionCallExtractor, FunctionDispatcher,
)
from flashllm.generation.structured_output import JSONSchema, JSONModeConstraint
from flashllm.eval.harness import EvalHarness, MMLUTask, HumanEvalTask, MTBenchTask, EvalResult
from flashllm.training.galore import GaLoreProjector, GaLoreAdamW
from flashllm.quantization.hqq import HQQQuantizer, HQQLinear
from flashllm.quantization.exl2 import EXL2LayerQuantizer, allocate_bits
from flashllm.export.gguf import GGUFWriter, GGMLType
from flashllm.training.sft import NEFTuneHook


class TestMoE:
    def test_router_output_shape(self):
        router = TopKRouter(hidden_size=64, num_experts=4, top_k=2)
        x = torch.randn(8, 64)
        weights, indices, lb_loss = router(x)
        assert weights.shape == (8, 2)
        assert indices.shape == (8, 2)
        assert lb_loss.dim() == 0

    def test_expert_forward(self):
        expert = Expert(hidden_size=64, intermediate_size=128)
        x = torch.randn(4, 64)
        out = expert(x)
        assert out.shape == (4, 64)

    def test_moe_layer_forward(self):
        moe = MoELayer(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        x = torch.randn(2, 10, 64)
        out, lb_loss = moe(x)
        assert out.shape == (2, 10, 64)
        assert lb_loss.item() >= 0

    def test_moe_transformer_block(self):
        block = MoETransformerBlock(
            hidden_size=64, num_heads=4, intermediate_size=128,
            num_experts=4, top_k=2,
        )
        x = torch.randn(1, 8, 64)
        out, present_kv, lb_loss = block(x)
        assert out.shape == (1, 8, 64)


class TestRoPEScaling:
    def test_linear_scaling(self):
        rope = LinearScaledRoPE(dim=64, max_position=2048, scaling_factor=2.0)
        x = torch.randn(1, 10, 64)
        cos, sin = rope(x, seq_len=100)
        assert cos.shape == (100, 64)

    def test_dynamic_ntk(self):
        rope = DynamicNTKScaledRoPE(dim=64, max_position=2048, scaling_factor=2.0)
        x = torch.randn(1, 10, 64)
        cos, sin = rope(x, seq_len=100)
        assert cos.shape == (100, 64)

    def test_yarn(self):
        rope = YaRNScaledRoPE(dim=64, max_position=2048, scaling_factor=2.0)
        x = torch.randn(1, 10, 64)
        cos, sin = rope(x, seq_len=100)
        assert cos.shape == (100, 64)

    def test_factory(self):
        for method in ["linear", "dynamic_ntk", "yarn"]:
            rope = get_rope_scaling(method, dim=64, scaling_factor=2.0)
            assert rope is not None

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_rope_scaling("invalid", dim=64)


class TestFunctionCalling:
    def test_schema_to_openai(self):
        schema = FunctionSchema(
            name="test_fn",
            description="A test",
            parameters=[FunctionParameter("x", "integer", "A number")],
        )
        openai = schema.to_openai_schema()
        assert openai["function"]["name"] == "test_fn"
        assert "x" in openai["function"]["parameters"]["properties"]

    def test_extract_tool_call(self):
        text = 'Some text <tool_call>{"name": "test", "arguments": {"x": 1}}</tool_call>'
        result = FunctionCallExtractor.extract(text)
        assert result is not None
        assert result["name"] == "test"

    def test_extract_json_block(self):
        text = '```json\n{"name": "test", "arguments": {}}\n```'
        result = FunctionCallExtractor.extract(text)
        assert result is not None

    def test_dispatcher(self):
        dispatcher = FunctionDispatcher()

        @dispatcher.register(name="add", description="Add numbers",
                             parameters=[FunctionParameter("a", "integer"), FunctionParameter("b", "integer")])
        def add(a, b):
            return a + b

        result = dispatcher.dispatch({"name": "add", "arguments": {"a": 3, "b": 4}})
        assert result == 7

    def test_dispatcher_missing(self):
        dispatcher = FunctionDispatcher()
        with pytest.raises(KeyError, match="not registered"):
            dispatcher.dispatch({"name": "missing"})


class TestGaLore:
    def test_projector(self):
        proj = GaLoreProjector(rank=4)
        grad = torch.randn(32, 64)
        projected = proj.project(grad)
        assert projected.shape[0] == 4 or projected.shape[1] == 4

    def test_project_back(self):
        proj = GaLoreProjector(rank=4)
        grad = torch.randn(32, 64)
        projected = proj.project(grad)
        reconstructed = proj.project_back(projected, grad.shape)
        assert reconstructed.shape == grad.shape

    def test_optimizer_step(self):
        model = torch.nn.Linear(32, 64)
        optimizer = GaLoreAdamW(
            [{"params": model.parameters(), "use_galore": True}],
            lr=0.01, rank=4,
        )
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()


class TestHQQ:
    def test_quantize_dequantize(self):
        quantizer = HQQQuantizer(bits=4, group_size=32)
        weight = torch.randn(64, 128)
        result = quantizer.quantize(weight)
        assert "q_weight" in result
        assert "scale" in result

        reconstructed = quantizer.dequantize(result)
        error = (weight - reconstructed).abs().mean()
        assert error < 1.0

    def test_hqq_linear(self):
        linear = torch.nn.Linear(32, 64, bias=False)
        hqq = HQQLinear.from_linear(linear, bits=4, group_size=16)
        x = torch.randn(2, 32)
        out = hqq(x)
        assert out.shape == (2, 64)


class TestEXL2:
    def test_layer_quantizer(self):
        quantizer = EXL2LayerQuantizer(bits=4.0, group_size=32)
        weight = torch.randn(64, 128)
        result = quantizer.quantize_weight(weight)
        assert "q_weight" in result
        assert result["q_weight"].shape == weight.shape

    def test_allocate_bits(self):
        sensitivities = {"layer1": 1.0, "layer2": 2.0, "layer3": 0.5}
        allocations = allocate_bits(sensitivities, target_bpw=4.0)
        assert len(allocations) == 3
        assert all(2.0 <= b <= 8.0 for b in allocations.values())


class TestEvalTasks:
    def test_mmlu_task(self):
        task = MMLUTask(subject="test")
        samples = task._synthetic_samples(5)
        assert len(samples) == 5
        prompt = task.format_prompt(samples[0])
        assert len(prompt) > 0

    def test_mmlu_scoring(self):
        task = MMLUTask()
        sample = {"answer": "A"}
        assert task.score(sample, "A")
        assert task.score(sample, "  A  ")
        assert not task.score(sample, "B")

    def test_humaneval_task(self):
        task = HumanEvalTask()
        samples = task._synthetic_samples(3)
        assert len(samples) == 3

    def test_mt_bench_task(self):
        task = MTBenchTask()
        samples = task.load_samples(5)
        assert len(samples) == 5

    def test_eval_result(self):
        result = EvalResult("test", score=0.8, num_samples=10, num_correct=8)
        assert "0.8" in repr(result)


class TestNEFTune:
    def test_hook_adds_noise(self):
        emb = torch.nn.Embedding(100, 64)
        hook = NEFTuneHook(noise_alpha=5.0)
        hook.register(emb)

        emb.train()
        ids = torch.tensor([[1, 2, 3]])
        out1 = emb(ids).clone()

        emb.train()
        out2 = emb(ids)
        assert not torch.allclose(out1, out2, atol=1e-6)

        hook.remove()

    def test_hook_noop_in_eval(self):
        emb = torch.nn.Embedding(100, 64)
        hook = NEFTuneHook(noise_alpha=5.0)
        hook.register(emb)

        emb.eval()
        ids = torch.tensor([[1, 2, 3]])
        out1 = emb(ids).clone()
        out2 = emb(ids)
        assert torch.allclose(out1, out2)
        hook.remove()


class TestGGUFWriter:
    def test_write_gguf(self, tmp_path):
        output = str(tmp_path / "test.gguf")
        writer = GGUFWriter(output, arch="llama")
        writer.add_uint32("llama.embedding_length", 64)
        writer.add_tensor("test.weight", torch.randn(32, 64), GGMLType.F16)
        writer.write()

        import os
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0
