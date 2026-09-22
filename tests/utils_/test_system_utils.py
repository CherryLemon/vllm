# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path

from vllm.utils.system_utils import _maybe_force_spawn, unique_filepath


def test_unique_filepath():
    temp_dir = tempfile.mkdtemp()
    path_fn = lambda i: Path(temp_dir) / f"file_{i}.txt"
    paths = set()
    for i in range(10):
        path = unique_filepath(path_fn)
        path.write_text("test")
        paths.add(path)
    assert len(paths) == 10
    assert len(list(Path(temp_dir).glob("*.txt"))) == 10


def test_numa_bind_forces_spawn(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "--numa-bind"])
    _maybe_force_spawn()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def _read_env_key(queue, key: str) -> None:
    queue.put(os.environ.get(key))


def test_spawn_inherits_env_removed_only_from_c_environ():
    """Spawn must see a key that Python still has after C ``unsetenv``."""
    import ctypes
    import multiprocessing

    key = "VLLM_TEST_SPAWN_ENV_SYNC"
    os.environ[key] = "1"
    try:
        rc = ctypes.CDLL(None).unsetenv(key.encode())
        assert rc == 0
        assert os.environ.get(key) == "1"
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_read_env_key, args=(queue, key))
        proc.start()
        try:
            assert queue.get(timeout=60) == "1"
        finally:
            proc.join(30)
            assert proc.exitcode == 0
    finally:
        os.environ.pop(key, None)
