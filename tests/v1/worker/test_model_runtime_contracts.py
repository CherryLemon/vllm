# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU contracts; these tests do not initialize an accelerator."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime():
    return load(
        "native_model_runtime_contract", ROOT / "vllm/v1/worker/model_runtime.py"
    )


@pytest.fixture
def proof():
    return load(
        "native_cpu_proof_contract",
        ROOT / "vllm/v1/attention/cpu_sequence_proof.py",
    )


def config(architecture, aliases=()):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture, hf_config=SimpleNamespace(architectures=aliases)
        )
    )


def test_unregistered_models_do_not_construct_a_runtime(runtime):
    assert runtime.create_model_runtime(config("Other")) is None


def test_registration_is_idempotent_and_model_scoped(runtime):
    calls = []
    provider = object()

    def factory(c):
        calls.append(c)
        return provider

    runtime.register_model_runtime(
        ("HYV4ForCausalLM", "HYV4VLForConditionalGeneration"), factory
    )
    runtime.register_model_runtime(("HYV4ForCausalLM",), factory)
    c = config("HYV4ForCausalLM", ("HYV4ForCausalLM",))
    assert runtime.create_model_runtime(c) is provider
    assert calls == [c]
    assert runtime.create_model_runtime(config("Other")) is None
    assert calls == [c]


def test_registration_failure_does_not_publish_partial_names(runtime):
    original = lambda c: "original"
    runtime.register_model_runtime(("Taken",), original)
    with pytest.raises(RuntimeError, match="already registered"):
        runtime.register_model_runtime(("Fresh", "Taken"), lambda c: "new")
    assert runtime.create_model_runtime(config("Fresh")) is None
    assert runtime.create_model_runtime(config("Taken")) == "original"


@pytest.mark.parametrize(
    "names,factory",
    [
        ([], lambda c: None),
        ([""], lambda c: None),
        (["A", "A"], lambda c: None),
        (["A"], None),
    ],
)
def test_invalid_registry_input_has_no_side_effect(runtime, names, factory):
    with pytest.raises(ValueError):
        runtime.register_model_runtime(names, factory)
    assert runtime.create_model_runtime(config("A")) is None


def test_factory_failure_propagates_without_retry(runtime):
    calls = []

    def fail(c):
        calls.append(c)
        raise RuntimeError("selected provider failed")

    runtime.register_model_runtime(("A",), fail)
    c = config("A")
    with pytest.raises(RuntimeError, match="selected provider failed"):
        runtime.create_model_runtime(c)
    assert calls == [c]


def test_conflicting_architecture_aliases_fail_before_factory(runtime):
    calls = []
    runtime.register_model_runtime(("A",), lambda c: calls.append("A"))
    runtime.register_model_runtime(("B",), lambda c: calls.append("B"))
    with pytest.raises(ValueError, match="different runtimes"):
        runtime.create_model_runtime(config("A", ("B",)))
    assert calls == []


@pytest.fixture
def current_step(proof):
    p = proof.CPUSequenceLengthsProof(
        "owner",
        3,
        ("decode", "prefill-a", "prefill-b"),
        (0, 4, 13, 21),
        (100, 8, 16),
        (20, 100, 100),
        (0, -1, 1),
        (3, -1, 0),
    )
    common = SimpleNamespace(
        cpu_seq_lens_proof=p,
        num_reqs=3,
        query_start_loc_cpu=torch.tensor(p.query_starts, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor((104, 17, 24), dtype=torch.int32),
        is_prefilling=torch.tensor((False, True, True), dtype=torch.bool),
    )
    chunk = SimpleNamespace(
        num_reqs=2,
        token_start=4,
        token_end=21,
        total_seq_lens=41,
        local_total_seq_lens=41,
        block_table=torch.empty(2, 8),
    )
    stager = SimpleNamespace(
        in_step=True,
        _cpu_proof_issuer="owner",
        _cpu_proof_generation=3,
        plan=SimpleNamespace(
            cpu_seq_lens_proof=p, num_reqs=3, total_tokens=21, req_ids=p.request_ids
        ),
    )
    return p, common, chunk, stager


def validate_step(proof, current_step):
    p, c, _, s = current_step
    return proof.validated_cpu_seq_lens_proof(
        s,
        p.request_ids,
        {"decode": 0, "prefill-b": 1},
        c.query_start_loc_cpu,
        c.seq_lens_cpu_upper_bound,
        c.is_prefilling,
        torch.tensor(p.prompt_lengths),
        3,
        3,
        21,
        21,
        False,
        None,
    )


def test_current_packet_qualifies_and_proof_is_immutable(proof, current_step):
    p, _, _, _ = current_step
    assert validate_step(proof, current_step) is p
    with pytest.raises(FrozenInstanceError):
        p.generation = 4


@pytest.mark.parametrize(
    "mutation",
    ["generation", "owner", "step", "request_order", "upper", "prompt", "prefilling"],
)
def test_packet_proof_rejects_one_changed_condition(proof, current_step, mutation):
    p, c, _, s = current_step
    if mutation == "generation":
        s._cpu_proof_generation += 1
    elif mutation == "owner":
        s._cpu_proof_issuer = "other"
    elif mutation == "step":
        s.in_step = False
    elif mutation == "request_order":
        s.plan.req_ids = p.request_ids[::-1]
    elif mutation == "upper":
        c.seq_lens_cpu_upper_bound[1] += 1
    elif mutation == "prompt":
        from dataclasses import replace

        s.plan.cpu_seq_lens_proof = replace(p, prompt_lengths=(20, 99, 100))
    elif mutation == "prefilling":
        c.is_prefilling[1] = False
    assert validate_step(proof, current_step) is None


def test_prefill_chunk_preserves_maximum_and_sum_distinction(proof, current_step):
    p, c, ch, _ = current_step
    bound = proof.validated_prefill_chunk_bound(c, ch, 1, 3, 1, 4, 1, 1, 1)
    assert bound.step_proof is p
    assert bound.seq_lens == (17, 24)
    assert max(bound.seq_lens) == 24
    assert sum(bound.seq_lens) == 41


@pytest.mark.parametrize(
    "mutation", ["draft", "sum", "table_rows", "token_range", "decode_request", "cp"]
)
def test_chunk_rejects_one_changed_condition(proof, current_step, mutation):
    p, c, ch, _ = current_step
    args = [c, ch, 1, 3, 1, 4, 1, 1, 1]
    if mutation == "draft":
        from dataclasses import replace

        c.cpu_seq_lens_proof = replace(p, previous_drafts=(3, -1, 1))
    elif mutation == "sum":
        ch.total_seq_lens += 1
    elif mutation == "table_rows":
        ch.block_table = torch.empty(1, 8)
    elif mutation == "token_range":
        ch.token_start = 3
    elif mutation == "decode_request":
        args[2] = 0
    elif mutation == "cp":
        args[-1] = 2
    assert proof.validated_prefill_chunk_bound(*args) is None


def test_step_context_invalidates_old_proof(proof, current_step):
    p, _, _, _ = current_step
    proof.set_current_cpu_seq_lens_proof(p)
    assert proof.get_current_cpu_seq_lens_proof() is p
    proof.set_current_cpu_seq_lens_proof(None)
    assert proof.get_current_cpu_seq_lens_proof() is None


def test_callable_objects_need_not_be_hashable(runtime):
    from dataclasses import dataclass

    @dataclass
    class Factory:
        value: str

        def __call__(self, config):
            return self.value

    factory = Factory("provider")
    runtime.register_model_runtime(("A",), factory)
    assert runtime.create_model_runtime(config("A", ("A",))) == "provider"
