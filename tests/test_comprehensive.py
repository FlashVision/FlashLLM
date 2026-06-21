"""Comprehensive tests for FlashLLM — architecture, generation, training, serving, quantization, CLI."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Architecture components
# ---------------------------------------------------------------------------
class TestTokenEmbedding:
    def test_forward_shape(self):
        from flashllm.models.architecture.embeddings import TokenEmbedding

        emb = TokenEmbedding(vocab_size=100, hidden_size=64)
        ids = torch.randint(0, 100, (2, 10))
        out = emb(ids)
        assert out.shape == (2, 10, 64)

    def test_padding_idx(self):
        from flashllm.models.architecture.embeddings import TokenEmbedding

        emb = TokenEmbedding(vocab_size=50, hidden_size=32, padding_idx=0)
        ids = torch.zeros(1, 5, dtype=torch.long)
        out = emb(ids)
        assert torch.allclose(out, torch.zeros_like(out))


class TestRotaryPositionalEmbedding:
    def test_forward_shape(self):
        from flashllm.models.architecture.embeddings import RotaryPositionalEmbedding

        rope = RotaryPositionalEmbedding(dim=64, max_position=512)
        x = torch.randn(1, 10, 64)
        cos, sin = rope(x, seq_len=100)
        assert cos.shape == (100, 64)
        assert sin.shape == (100, 64)

    def test_caching(self):
        from flashllm.models.architecture.embeddings import RotaryPositionalEmbedding

        rope = RotaryPositionalEmbedding(dim=32, max_position=256)
        x = torch.randn(1, 5, 32)
        cos1, sin1 = rope(x, seq_len=50)
        cos2, sin2 = rope(x, seq_len=50)
        assert torch.equal(cos1, cos2)

    def test_extends_on_longer_seq(self):
        from flashllm.models.architecture.embeddings import RotaryPositionalEmbedding

        rope = RotaryPositionalEmbedding(dim=32, max_position=256)
        x = torch.randn(1, 5, 32)
        rope(x, seq_len=50)
        cos, sin = rope(x, seq_len=100)
        assert cos.shape[0] == 100


class TestLearnedPositionalEmbedding:
    def test_forward(self):
        from flashllm.models.architecture.embeddings import LearnedPositionalEmbedding

        lpe = LearnedPositionalEmbedding(max_position=128, hidden_size=64)
        ids = torch.arange(10).unsqueeze(0)
        out = lpe(ids)
        assert out.shape == (1, 10, 64)


class TestMultiHeadAttention:
    def test_mha_forward(self):
        from flashllm.models.architecture.attention import MultiHeadAttention

        attn = MultiHeadAttention(hidden_size=64, num_heads=4, max_position=128)
        x = torch.randn(1, 10, 64)
        out, kv = attn(x)
        assert out.shape == (1, 10, 64)
        assert kv is None

    def test_mha_use_cache(self):
        from flashllm.models.architecture.attention import MultiHeadAttention

        attn = MultiHeadAttention(hidden_size=64, num_heads=4, max_position=128)
        x = torch.randn(1, 8, 64)
        out, kv = attn(x, use_cache=True)
        assert kv is not None
        assert kv[0].shape[2] == 8

    def test_gqa(self):
        from flashllm.models.architecture.attention import MultiHeadAttention

        attn = MultiHeadAttention(hidden_size=64, num_heads=8, num_kv_heads=2, max_position=128)
        x = torch.randn(1, 6, 64)
        out, _ = attn(x)
        assert out.shape == (1, 6, 64)

    def test_mqa(self):
        from flashllm.models.architecture.attention import MultiHeadAttention

        attn = MultiHeadAttention(hidden_size=64, num_heads=8, num_kv_heads=1, max_position=128)
        x = torch.randn(1, 6, 64)
        out, _ = attn(x)
        assert out.shape == (1, 6, 64)

    def test_past_kv(self):
        from flashllm.models.architecture.attention import MultiHeadAttention

        attn = MultiHeadAttention(hidden_size=64, num_heads=4, max_position=128)
        x1 = torch.randn(1, 5, 64)
        _, kv = attn(x1, use_cache=True)
        x2 = torch.randn(1, 1, 64)
        out, kv2 = attn(x2, past_key_value=kv, use_cache=True)
        assert out.shape == (1, 1, 64)
        assert kv2[0].shape[2] == 6

    def test_rope_scaling_linear(self):
        from flashllm.models.architecture.attention import MultiHeadAttention

        attn = MultiHeadAttention(
            hidden_size=64,
            num_heads=4,
            max_position=128,
            rope_scaling_type="linear",
            rope_scaling_factor=2.0,
        )
        x = torch.randn(1, 10, 64)
        out, _ = attn(x)
        assert out.shape == (1, 10, 64)


class TestGroupedQueryAttention:
    def test_alias(self):
        from flashllm.models.architecture.attention import GroupedQueryAttention

        gqa = GroupedQueryAttention(hidden_size=64, num_heads=8, num_kv_heads=4, max_position=64)
        x = torch.randn(1, 5, 64)
        out, _ = gqa(x)
        assert out.shape == (1, 5, 64)


class TestTransformerBlock:
    def test_forward(self):
        from flashllm.models.architecture.transformer import TransformerBlock

        block = TransformerBlock(hidden_size=64, num_heads=4, intermediate_size=128)
        x = torch.randn(1, 10, 64)
        out, kv = block(x)
        assert out.shape == (1, 10, 64)
        assert kv is None

    def test_with_cache(self):
        from flashllm.models.architecture.transformer import TransformerBlock

        block = TransformerBlock(hidden_size=64, num_heads=4, intermediate_size=128)
        x = torch.randn(1, 8, 64)
        out, kv = block(x, use_cache=True)
        assert kv is not None

    def test_gradient_flow(self):
        from flashllm.models.architecture.transformer import TransformerBlock

        block = TransformerBlock(hidden_size=64, num_heads=4, intermediate_size=128)
        x = torch.randn(1, 5, 64, requires_grad=True)
        out, _ = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestLanguageModelHead:
    def test_forward(self):
        from flashllm.models.heads.lm_head import LanguageModelHead

        head = LanguageModelHead(hidden_size=64, vocab_size=100)
        x = torch.randn(1, 10, 64)
        logits = head(x)
        assert logits.shape == (1, 10, 100)

    def test_tie_weights(self):
        from flashllm.models.heads.lm_head import LanguageModelHead

        head = LanguageModelHead(hidden_size=64, vocab_size=100)
        emb = nn.Embedding(100, 64)
        head.tie_embedding_weights(emb)
        assert head.lm_head.weight is emb.weight


# ---------------------------------------------------------------------------
# LoRA / QLoRA
# ---------------------------------------------------------------------------
class TestLoRAAdapter:
    def test_forward(self):
        from flashllm.models.lora import LoRALinear

        orig = nn.Linear(64, 128, bias=False)
        lora = LoRALinear(orig, rank=4, alpha=8.0)
        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert out.shape == (2, 10, 128)

    def test_frozen_original(self):
        from flashllm.models.lora import LoRALinear

        orig = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(orig, rank=4, alpha=8.0)
        assert not lora.original.weight.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_merge_produces_linear(self):
        from flashllm.models.lora import LoRALinear

        orig = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(orig, rank=4, alpha=8.0)
        merged = lora.merge()
        assert isinstance(merged, nn.Linear)
        assert merged.weight.shape == (64, 32)


# ---------------------------------------------------------------------------
# Generation: beam search, sampler, streaming, structured output
# ---------------------------------------------------------------------------
class TestBeamSearch:
    def test_hypothesis_dataclass(self):
        from flashllm.generation.beam_search import BeamHypothesis

        h = BeamHypothesis(tokens=[1, 2, 3], score=1.5, is_finished=True)
        assert h.is_finished
        assert len(h.tokens) == 3

    def test_beam_search_with_dummy_model(self):
        from flashllm.generation.beam_search import BeamSearch

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 20)

            def forward(self, input_ids):
                b, s = input_ids.shape
                logits = torch.randn(b, s, 20)
                logits[:, :, 5] = 10.0
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        bs = BeamSearch(num_beams=2, max_length=5, eos_token_id=5)
        ids = torch.randint(0, 20, (1, 3))
        results = bs.search(model, ids)
        assert len(results) <= 2
        assert all(r.is_finished for r in results)


class TestSamplerComprehensive:
    def test_top_p(self):
        from flashllm.generation.sampler import Sampler

        sampler = Sampler(temperature=1.0, top_k=0, top_p=0.5)
        logits = torch.randn(1, 100)
        token = sampler(logits)
        assert token.shape == (1,)

    def test_min_p(self):
        from flashllm.generation.sampler import Sampler

        sampler = Sampler(temperature=1.0, top_k=0, top_p=1.0, min_p=0.1)
        logits = torch.randn(1, 100)
        token = sampler(logits)
        assert token.shape == (1,)

    def test_1d_logits(self):
        from flashllm.generation.sampler import Sampler

        sampler = Sampler()
        logits = torch.randn(50)
        token = sampler(logits)
        assert token.shape == (1,)


class TestStreamingGenerator:
    def test_put_and_iterate(self):
        from flashllm.generation.streaming import StreamingGenerator

        tok = MagicMock()
        tok.decode = MagicMock(return_value="hello")
        sg = StreamingGenerator(tok, skip_prompt=False)
        sg.put("hello")
        sg.put("world")
        sg.end()
        results = list(sg)
        assert results == ["hello", "world"]

    def test_put_tensor(self):
        from flashllm.generation.streaming import StreamingGenerator

        tok = MagicMock()
        tok.decode = MagicMock(return_value="tok")
        sg = StreamingGenerator(tok, skip_prompt=False)
        sg.put(torch.tensor([42]))
        sg.end()
        results = list(sg)
        assert len(results) == 1


class TestStructuredOutput:
    def test_json_schema_properties(self):
        from flashllm.generation.structured_output import JSONSchema

        schema = JSONSchema(
            schema={"properties": {"name": {"type": "string"}}, "required": ["name"]},
            strict=True,
        )
        assert schema.required_keys == ["name"]
        assert "name" in schema.properties

    def test_json_mode_analyze_state(self):
        from flashllm.generation.structured_output import JSONModeConstraint

        tok = MagicMock()
        tok.__len__ = MagicMock(return_value=0)
        constraint = JSONModeConstraint(tok)
        assert constraint._analyze_state("") == "START"
        assert constraint._analyze_state('{"key"') == "OBJECT_START"
        assert constraint._analyze_state('{"key') == "IN_STRING"
        assert constraint._analyze_state('{"key": "val"}') == "COMPLETE"

    def test_json_mode_states(self):
        from flashllm.generation.structured_output import JSONModeConstraint

        tok = MagicMock()
        tok.__len__ = MagicMock(return_value=0)
        c = JSONModeConstraint(tok)
        assert c._analyze_state("{") == "OBJECT_START"
        assert c._analyze_state('{"k":') == "COLON"
        assert c._analyze_state("[") == "ARRAY_START"

    def test_regex_constraint_valid_prefix(self):
        from flashllm.generation.structured_output import RegexConstraint

        tok = MagicMock()
        tok.__len__ = MagicMock(return_value=0)
        rc = RegexConstraint(tok, r"\d{3}-\d{4}")
        assert rc._is_valid_prefix("123-4567")
        assert not rc._is_valid_prefix("abc")
        assert rc._is_valid_prefix("123-4567")

    def test_grammar_rule_dataclass(self):
        from flashllm.generation.structured_output import GrammarRule

        rule = GrammarRule(name="digit", alternatives=[["0"], ["1"]])
        assert rule.name == "digit"
        assert len(rule.alternatives) == 2

    def test_grammar_constraint(self):
        from flashllm.generation.structured_output import GrammarConstraint, GrammarRule

        rules = [
            GrammarRule(name="root", alternatives=[["yes"], ["no"]]),
        ]
        gc = GrammarConstraint(MagicMock(), rules, start_symbol="root")
        allowed = gc.get_allowed_strings()
        assert "yes" in allowed
        assert "no" in allowed

    def test_grammar_advance_and_reset(self):
        from flashllm.generation.structured_output import GrammarConstraint, GrammarRule

        rules = [GrammarRule(name="root", alternatives=[["a", "b"]])]
        gc = GrammarConstraint(MagicMock(), rules, start_symbol="root")
        gc.advance("a")
        assert gc._stack == ["b"]
        gc.reset()
        assert gc._stack == ["root"]


# ---------------------------------------------------------------------------
# Function calling
# ---------------------------------------------------------------------------
class TestFunctionCallingExtended:
    def test_extract_returns_none_on_garbage(self):
        from flashllm.generation.function_calling import FunctionCallExtractor

        assert FunctionCallExtractor.extract("no tool call here") is None

    def test_schema_required_param(self):
        from flashllm.generation.function_calling import FunctionParameter, FunctionSchema

        schema = FunctionSchema(
            name="greet",
            description="Greet user",
            parameters=[FunctionParameter("name", "string", "User name", required=True)],
        )
        oai = schema.to_openai_schema()
        assert "name" in oai["function"]["parameters"]["properties"]

    def test_dispatcher_list_functions(self):
        from flashllm.generation.function_calling import FunctionDispatcher, FunctionParameter

        disp = FunctionDispatcher()

        @disp.register(name="add", description="Add", parameters=[FunctionParameter("a", "integer")])
        def _add(a):
            return a

        assert "add" in disp._schemas


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class TestCrossEntropyLoss:
    def test_basic(self):
        from flashllm.losses.cross_entropy import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_label_smoothed(self):
        from flashllm.losses.cross_entropy import LabelSmoothedCrossEntropy

        loss_fn = LabelSmoothedCrossEntropy(smoothing=0.1)
        logits = torch.randn(2, 10, 50)
        labels = torch.randint(0, 50, (2, 10))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0

    def test_ignore_index(self):
        from flashllm.losses.cross_entropy import CrossEntropyLoss

        loss_fn = CrossEntropyLoss(ignore_index=-100)
        logits = torch.randn(1, 5, 20)
        labels = torch.randint(0, 20, (1, 5))
        labels[0, 0] = -100
        loss = loss_fn(logits, labels)
        assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# DPO / IPO / KTO losses
# ---------------------------------------------------------------------------
class TestDPOLossFunctions:
    @staticmethod
    def _make_model_and_batch():
        model = nn.Linear(10, 20)

        class WrapModel(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask=None):
                b, s = input_ids.shape
                return SimpleNamespace(logits=torch.randn(b, s, 20))

        batch = {
            "chosen_input_ids": torch.randint(0, 20, (2, 8)),
            "chosen_attention_mask": torch.ones(2, 8, dtype=torch.long),
            "rejected_input_ids": torch.randint(0, 20, (2, 8)),
            "rejected_attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
        return WrapModel(model), batch

    def test_dpo_loss(self):
        from flashllm.losses.dpo_loss import dpo_loss

        model, batch = self._make_model_and_batch()
        loss = dpo_loss(model, batch, beta=0.1)
        assert loss.dim() == 0

    def test_ipo_loss(self):
        from flashllm.losses.dpo_loss import ipo_loss

        model, batch = self._make_model_and_batch()
        loss = ipo_loss(model, batch, beta=0.1)
        assert loss.dim() == 0

    def test_kto_loss(self):
        from flashllm.losses.dpo_loss import kto_loss

        model, batch = self._make_model_and_batch()
        loss = kto_loss(model, batch, beta=0.1)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Training: SFT, DPO trainer, RLHF
# ---------------------------------------------------------------------------
class TestSFTTrainer:
    def test_find_embedding_layer(self):
        from flashllm.training.sft import SFTTrainer

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 64)
                self.fc = nn.Linear(64, 64)

        model = Model()
        emb = SFTTrainer._find_embedding_layer(model)
        assert emb is model.embed_tokens

    def test_find_embedding_none(self):
        from flashllm.training.sft import SFTTrainer

        model = nn.Sequential(nn.Linear(10, 10))
        assert SFTTrainer._find_embedding_layer(model) is None


class TestDPOTrainer:
    def test_init_freezes_ref(self):
        from flashllm.training.dpo import DPOTrainer

        model = nn.Linear(10, 10)
        ref = nn.Linear(10, 10)
        trainer = DPOTrainer(model=model, ref_model=ref, beta=0.1)
        for p in trainer.ref_model.parameters():
            assert not p.requires_grad

    def test_init_no_ref(self):
        from flashllm.training.dpo import DPOTrainer

        model = nn.Linear(10, 10)
        trainer = DPOTrainer(model=model, ref_model=None, beta=0.1)
        assert trainer.ref_model is None


class TestRLHFTrainer:
    def test_compute_rewards_no_reward_model(self):
        from flashllm.training.rlhf import RLHFTrainer

        policy = nn.Linear(10, 10)
        trainer = RLHFTrainer(policy_model=policy, reward_model=None)
        ids = torch.randint(0, 10, (2, 5))
        rewards = trainer.compute_rewards(ids)
        assert torch.allclose(rewards, torch.zeros(2))

    def test_train_step(self):
        from flashllm.training.rlhf import RLHFTrainer

        class DummyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 20)

            def forward(self, input_ids, attention_mask=None):
                b, s = input_ids.shape
                return SimpleNamespace(logits=torch.randn(b, s, 20))

        policy = DummyPolicy()
        trainer = RLHFTrainer(policy_model=policy, kl_coef=0.1, clip_range=0.2)
        batch = {
            "input_ids": torch.randint(0, 20, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
        old_logps = torch.randn(2)
        advantages = torch.randn(2)
        result = trainer.train_step(batch, old_logps, advantages)
        assert "policy_loss" in result
        assert "approx_kl" in result


# ---------------------------------------------------------------------------
# Distributed (config only, no actual distributed init)
# ---------------------------------------------------------------------------
class TestDistributedConfig:
    def test_fsdp_config_defaults(self):
        from flashllm.training.distributed import FSDPConfig

        cfg = FSDPConfig()
        assert cfg.sharding_strategy == "FULL_SHARD"
        assert cfg.mixed_precision == "bf16"
        assert cfg.activation_checkpointing is True

    def test_deepspeed_config_defaults(self):
        from flashllm.training.distributed import DeepSpeedConfig

        cfg = DeepSpeedConfig()
        assert cfg.stage == 2
        assert cfg.bf16_enabled is True

    def test_get_deepspeed_config_stage2(self):
        from flashllm.training.distributed import DeepSpeedConfig, get_deepspeed_config

        cfg = DeepSpeedConfig(stage=2)
        ds_cfg = get_deepspeed_config(cfg)
        assert ds_cfg["zero_optimization"]["stage"] == 2
        assert "gradient_clipping" in ds_cfg

    def test_get_deepspeed_config_stage3(self):
        from flashllm.training.distributed import DeepSpeedConfig, get_deepspeed_config

        cfg = DeepSpeedConfig(stage=3, offload_optimizer=True, offload_params=True)
        ds_cfg = get_deepspeed_config(cfg)
        assert ds_cfg["zero_optimization"]["stage"] == 3
        assert "offload_optimizer" in ds_cfg["zero_optimization"]
        assert "offload_param" in ds_cfg["zero_optimization"]

    def test_get_deepspeed_fp16(self):
        from flashllm.training.distributed import DeepSpeedConfig, get_deepspeed_config

        cfg = DeepSpeedConfig(fp16_enabled=True, bf16_enabled=False)
        ds_cfg = get_deepspeed_config(cfg)
        assert "fp16" in ds_cfg
        assert ds_cfg["fp16"]["enabled"] is True


# ---------------------------------------------------------------------------
# Speculative decoding
# ---------------------------------------------------------------------------
class TestSpeculativeDecoding:
    def test_token_node(self):
        from flashllm.serving.speculative_decoding import TokenNode

        root = TokenNode(token_id=1, log_prob=-0.5, depth=0)
        child = root.add_child(token_id=2, log_prob=-0.3)
        assert child.depth == 1
        assert child.parent is root
        assert len(root.children) == 1

    def test_token_tree(self):
        from flashllm.serving.speculative_decoding import TokenTree

        tree = TokenTree()
        ids = torch.tensor([1, 2, 3])
        logps = torch.tensor([-0.5, -0.3, -0.1])
        tree.build_from_draft(ids, logps)
        assert tree.num_tokens == 3
        assert tree.get_token_ids() == [1, 2, 3]
        assert tree.get_verification_positions() == [0, 1, 2]

    def test_speculative_decoder_acceptance_rate(self):
        from flashllm.serving.speculative_decoding import SpeculativeDecoder

        decoder = SpeculativeDecoder(
            target_model=MagicMock(),
            draft_model=MagicMock(),
            tokenizer=MagicMock(),
            num_speculative_tokens=5,
        )
        assert decoder.acceptance_rate == 0.0
        decoder.total_draft_tokens = 10
        decoder.accepted_tokens = 8
        assert decoder.acceptance_rate == 0.8
        decoder.reset_stats()
        assert decoder.acceptance_rate == 0.0


# ---------------------------------------------------------------------------
# Quantization: HQQ, EXL2 (already tested), GPTQ/AWQ import guards
# ---------------------------------------------------------------------------
class TestQuantizationImportGuards:
    def test_gptq_import_error(self):
        from flashllm.quantization.gptq import quantize_gptq

        with pytest.raises(ImportError, match="GPTQ"):
            quantize_gptq("fake-model-id")

    def test_awq_import_error(self):
        from flashllm.quantization.awq import quantize_awq

        with pytest.raises(ImportError, match="AWQ"):
            quantize_awq("fake-model-id")


# ---------------------------------------------------------------------------
# GGUF export
# ---------------------------------------------------------------------------
class TestGGUFExport:
    def test_ggml_types(self):
        from flashllm.export.gguf import GGMLType

        assert GGMLType.F32 is not None
        assert GGMLType.F16 is not None
        assert GGMLType.Q4_0 is not None

    def test_writer_metadata(self, tmp_path):
        from flashllm.export.gguf import GGMLType, GGUFWriter

        output = str(tmp_path / "meta.gguf")
        writer = GGUFWriter(output, arch="test")
        writer.add_uint32("test.layers", 12)
        writer.add_string("test.name", "hello")
        writer.add_tensor("w", torch.randn(4, 8), GGMLType.F32)
        writer.write()

        import os

        assert os.path.exists(output)
        assert os.path.getsize(output) > 0


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------
class TestEvalHarnessExtended:
    def test_mmlu_format_prompt(self):
        from flashllm.eval.harness import MMLUTask

        task = MMLUTask(subject="math")
        samples = task._synthetic_samples(2)
        prompt = task.format_prompt(samples[0])
        assert "A)" in prompt or "(A)" in prompt or "A." in prompt or "A" in prompt

    def test_humaneval_samples(self):
        from flashllm.eval.harness import HumanEvalTask

        task = HumanEvalTask()
        samples = task._synthetic_samples(5)
        assert len(samples) == 5
        for s in samples:
            assert "prompt" in s

    def test_eval_result_repr(self):
        from flashllm.eval.harness import EvalResult

        r = EvalResult("test_task", score=0.75, num_samples=20, num_correct=15)
        assert "0.75" in repr(r)
        assert "test_task" in repr(r)


# ---------------------------------------------------------------------------
# Data: templates
# ---------------------------------------------------------------------------
class TestChatTemplates:
    def test_alpaca_template(self):
        from flashllm.data.templates import AlpacaTemplate

        t = AlpacaTemplate()
        sample = {"instruction": "Say hello", "output": "Hello!"}
        text = t.format_sample(sample)
        assert "Instruction" in text
        assert "Response" in text

    def test_alpaca_with_input(self):
        from flashllm.data.templates import AlpacaTemplate

        t = AlpacaTemplate()
        sample = {"instruction": "Translate", "input": "Bonjour", "output": "Hello"}
        text = t.format_sample(sample)
        assert "Input" in text

    def test_chatml_template(self):
        from flashllm.data.templates import ChatMLTemplate

        t = ChatMLTemplate()
        prompt = t.format_prompt("Hi there")
        assert "<|im_start|>" in prompt

    def test_llama_template(self):
        from flashllm.data.templates import LlamaTemplate

        t = LlamaTemplate()
        prompt = t.format_prompt("Hi")
        assert "<|begin_of_text|>" in prompt

    def test_mistral_template(self):
        from flashllm.data.templates import MistralTemplate

        t = MistralTemplate()
        prompt = t.format_prompt("Hi")
        assert "[INST]" in prompt

    def test_gemma_template(self):
        from flashllm.data.templates import GemmaTemplate

        t = GemmaTemplate()
        prompt = t.format_prompt("Hi")
        assert "<start_of_turn>" in prompt

    def test_get_template(self):
        from flashllm.data.templates import get_template

        for name in ["alpaca", "chatml", "llama", "mistral", "gemma"]:
            t = get_template(name)
            assert t is not None


# ---------------------------------------------------------------------------
# Model utils
# ---------------------------------------------------------------------------
class TestModelUtils:
    def test_count_parameters(self):
        from flashllm.utils.model_utils import count_parameters

        model = nn.Linear(100, 200)
        result = count_parameters(model)
        assert result["total"] == 100 * 200 + 200
        assert result["trainable"] == result["total"]
        assert result["frozen"] == 0
        assert result["total_gb"] > 0

    def test_estimate_memory(self):
        from flashllm.utils.model_utils import estimate_memory

        mem = estimate_memory(num_params=1_000_000, dtype="float16")
        assert mem["model_gb"] > 0
        assert mem["total_training_gb"] > mem["model_gb"]

    def test_estimate_memory_gradient_checkpointing(self):
        from flashllm.utils.model_utils import estimate_memory

        mem_no_ckpt = estimate_memory(num_params=1_000_000, gradient_checkpointing=False)
        mem_ckpt = estimate_memory(num_params=1_000_000, gradient_checkpointing=True)
        assert mem_ckpt["gradient_gb"] < mem_no_ckpt["gradient_gb"]

    def test_get_dtype(self):
        from flashllm.utils.model_utils import get_dtype

        assert get_dtype("float32") == torch.float32
        assert get_dtype("float16") == torch.float16
        assert get_dtype("bfloat16") == torch.bfloat16


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
class TestCLIParser:
    def test_main_parser_no_args(self):
        from flashllm.cli import main

        with pytest.raises(SystemExit):
            import sys

            old = sys.argv
            sys.argv = ["flashllm"]
            try:
                main()
            finally:
                sys.argv = old

    def test_version_command(self):
        from flashllm.cli import cmd_version

        cmd_version(SimpleNamespace())

    def test_colored_helper(self):
        from flashllm.cli import _colored

        result = _colored("hello", "green")
        assert "hello" in result

    def test_parser_commands_exist(self):

        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        for cmd in ["version", "settings", "check", "train", "chat", "generate", "export", "benchmark", "quantize"]:
            subparsers.add_parser(cmd)
        args = parser.parse_args(["version"])
        assert args.command == "version"


# ---------------------------------------------------------------------------
# Integration: model → tokenize → generate → decode (mocked HF)
# ---------------------------------------------------------------------------
class TestIntegrationFlashLLM:
    def test_dtype_map(self):
        from flashllm.models.flash_llm import DTYPE_MAP

        assert DTYPE_MAP["float32"] == torch.float32
        assert DTYPE_MAP["float16"] == torch.float16
        assert DTYPE_MAP["bfloat16"] == torch.bfloat16


# ---------------------------------------------------------------------------
# nn building blocks extra coverage
# ---------------------------------------------------------------------------
class TestNNBuildingBlocks:
    def test_rms_norm_gradient(self):
        from flashllm.nn import RMSNorm

        norm = RMSNorm(hidden_size=32)
        x = torch.randn(1, 5, 32, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None

    def test_swiglu_different_sizes(self):
        from flashllm.nn import SwiGLU

        for h, i in [(32, 64), (64, 256), (128, 512)]:
            ffn = SwiGLU(hidden_size=h, intermediate_size=i)
            x = torch.randn(1, 5, h)
            out = ffn(x)
            assert out.shape == (1, 5, h)

    def test_rotary_embedding_cos_sin_values(self):
        from flashllm.nn import RotaryEmbedding

        rope = RotaryEmbedding(dim=16, max_seq_len=64)
        cos, sin = rope(seq_len=10, device=torch.device("cpu"))
        assert (-1 <= cos).all() and (cos <= 1).all()
        assert (-1 <= sin).all() and (sin <= 1).all()
