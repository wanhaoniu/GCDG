from __future__ import annotations

import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator


def patch_numpy_legacy_aliases() -> None:
    """兼容旧仓库对 numpy 过时别名的依赖。

    注意:
    - 这里只在 wrapper 内补丁，不修改 AnyGrasp / SuctionNet 原仓库代码。
    - 通过检查 numpy.__dict__ 避免触发新版本 numpy 的 FutureWarning。
    """

    import numpy as np

    legacy_aliases = {
        "float": float,
        "int": int,
        "bool": np.bool_,
    }

    for alias_name, alias_value in legacy_aliases.items():
        if alias_name not in np.__dict__:
            setattr(np, alias_name, alias_value)


def ensure_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


@contextmanager
def temporary_cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(previous))


@contextmanager
def suppress_output() -> Iterator[None]:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


@contextmanager
def patch_torch_load_cpu_checkpoint_compat() -> Iterator[None]:
    """兼容 checkpoint 保存在 CUDA，但当前运行环境没有可用 CUDA 的场景。

    设计目标：
    - 不修改 AnyGrasp 原仓库 / 二进制。
    - 仅在 wrapper 调用期间临时补丁 torch.load。
    """

    try:
        import torch
    except Exception:
        yield
        return

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if cuda_available:
        yield
        return

    original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = torch.device("cpu")
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load
