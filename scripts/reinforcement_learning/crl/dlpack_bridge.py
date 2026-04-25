# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zero-copy tensor bridge between PyTorch (IsaacLab) and JAX (scaling-crl).

The CRL training stack is JAX/Flax; IsaacLab produces torch GPU tensors. Copying
arrays across that boundary per env-step would shred throughput, so we use DLPack
to pass the underlying CUDA buffer between frameworks without allocation.

Usage:

.. code-block:: python

    from dlpack_bridge import configure_jax_memory, torch_to_jax, jax_to_torch

    # Must be called *before* any ``import jax`` / before JAX initializes CUDA.
    configure_jax_memory(mem_fraction=0.3)

    import torch
    import jax

    x = torch.randn(1024, 32, device="cuda")
    xj = torch_to_jax(x)  # zero-copy; xj lives on the same CUDA device
    yj = xj * 2.0  # JAX op
    yt = jax_to_torch(yj)  # zero-copy back
    assert yt.device == x.device

Notes:
    - JAX by default pre-allocates ~90 percent of GPU memory on import. Leaving
      IsaacLab + PhysX insufficient headroom causes silent OOMs. Call
      :func:`configure_jax_memory` before the first ``import jax`` to fix.
    - The conversion is zero-copy but the semantics are "borrow, not take": the
      consumer must not mutate the buffer after the producer has released it.
      In practice we do not mutate in place on either side; for safety we call
      ``.detach().contiguous()`` on torch inputs and ``jnp.asarray(...).copy()``
      is avoided.
    - For bidirectional conversion of int tensors on CUDA, note DLPack requires
      native C-contiguous layout; non-contiguous views must be materialized.
"""

from __future__ import annotations

import os
import warnings


def configure_jax_memory(mem_fraction: float = 0.3, preallocate: bool = False) -> None:
    """Configure JAX GPU memory behavior.

    Must be called *before* JAX is imported. Sets environment variables consumed
    by the XLA GPU client at startup.

    Args:
        mem_fraction: Fraction of per-GPU memory JAX is allowed to use. IsaacLab +
            PhysX typically need 40–70% of a 24 GB GPU for physics/state; we want
            JAX below that so the two stacks coexist. Empirically 0.3 is safe on
            an A6000 (48 GB), and may need to drop to 0.2 on 24 GB cards.
        preallocate: If ``True``, JAX grabs the entire ``mem_fraction`` at import
            time (avoids fragmentation). If ``False`` (default), JAX allocates
            lazily. False is safer for coexistence with PyTorch's caching allocator.
    """
    if "jax" in _imported_modules():
        warnings.warn(
            "configure_jax_memory called after JAX was imported; JAX env vars are read "
            "at import time and will have no effect now. Import order bug upstream.",
            RuntimeWarning,
            stacklevel=2,
        )
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if preallocate else "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem_fraction:.3f}"


def _imported_modules() -> set[str]:
    import sys

    return set(sys.modules)


def torch_to_jax(t):
    """Zero-copy convert a torch tensor to a JAX array.

    Args:
        t: A torch tensor, typically on CUDA. Forced to contiguous if needed.

    Returns:
        A JAX array sharing the same memory as ``t``. Device is preserved
        (CUDA tensors -> JAX GPU arrays; CPU tensors -> JAX CPU arrays).
    """
    import jax
    import torch.utils.dlpack as torch_dlpack

    t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    # The legacy ``jax.dlpack.from_dlpack(capsule)`` path is deprecated in newer
    # JAX; the modern API is ``jax.dlpack.from_dlpack(arr)`` where ``arr`` is any
    # object implementing ``__dlpack__``. Torch tensors have that since 2.0.
    try:
        return jax.dlpack.from_dlpack(t)
    except Exception:
        return jax.dlpack.from_dlpack(torch_dlpack.to_dlpack(t))


def jax_to_torch(a):
    """Zero-copy convert a JAX array to a torch tensor.

    Args:
        a: A JAX array.

    Returns:
        A torch tensor sharing the same memory as ``a``. Device is preserved.
    """
    import jax
    import torch
    import torch.utils.dlpack as torch_dlpack

    try:
        return torch.from_dlpack(a)
    except Exception:
        capsule = jax.dlpack.to_dlpack(a)
        return torch_dlpack.from_dlpack(capsule)
