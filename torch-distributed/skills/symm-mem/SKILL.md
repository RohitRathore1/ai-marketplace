---
name: symm-mem
description: Reference for PyTorch symmetric memory — the low-level P2P GPU communication substrate used by async tensor parallelism (TP) and sequence parallelism. Covers the _SymmetricMemory C++ binding (empty_strided_p2p, rendezvous, get_buffer, barrier, has_multicast_support, signal_pad_size), the Python user-facing API (symm_mem.empty, rendezvous, get_symm_mem_workspace, set_backend, set_signal_pad_size), the fused operator library (fused_all_gather_matmul, fused_matmul_reduce_scatter, fused_all_gather_scaled_matmul, fused_scaled_matmul_reduce_scatter, _low_contention_all_gather, _low_contention_all_gather_ce_multicast, _low_contention_reduce_scatter, get_remote_tensors, all_to_all_nd), the fused-op dispatch decision tree (native _async_input_mm path, multimem/NVLS path, pipelined impl, fallback), the CUDA/NCCL/NVSHMEM backends, and the signal pad layout. Use as the knowledge base for implementing or debugging async TP, sequence parallel (Ulysses-style), custom P2P collectives, CUDA-graph-safe workspace management, and multicast collective operations.
---

# Symmetric Memory

Reference for `torch.distributed._symmetric_memory` — PyTorch's P2P GPU communication layer that powers async tensor parallelism and sequence parallelism.

## Where Symm Mem Fits

```
Training forward pass (tensor parallel)
         |
         v
  ┌──────────────────────────────────────────────────┐
  │  Async TP (fused_all_gather_matmul)              │
  │  ┌─────────────┐    ┌─────────────┐              │
  │  │ all-gather  │ ── │   matmul    │  pipelined   │
  │  │  (symm_mem) │    │  (cublas)   │  overlap     │
  │  └─────────────┘    └─────────────┘              │
  └──────────────────────────────────────────────────┘
         |
         v
  ┌──────────────────────────────────────────────────┐
  │  Async TP (fused_matmul_reduce_scatter)          │
  │  ┌─────────────┐    ┌─────────────┐              │
  │  │   matmul    │ ── │reduce-scatter│  pipelined   │
  │  │  (cublas)   │    │  (symm_mem) │  overlap     │
  │  └─────────────┘    └─────────────┘              │
  └──────────────────────────────────────────────────┘
         |
         v
  Sequence Parallel (all_to_all_nd)
  Low-contention collectives (_low_contention_all_gather/_low_contention_reduce_scatter)
  Custom P2P (get_remote_tensors)
```

Symm mem bypasses NCCL kernel launches by using direct GPU P2P reads/writes into memory that is simultaneously mapped on all ranks in an NVLink domain. The fused ops overlap communication with computation on a dedicated backend stream.

## Progressive Disclosure

| Your Goal | Read This | What It Provides |
|---|---|---|
| Understand allocator lifecycle, C++ bindings, signal pad | [ARCHITECTURE.md](ARCHITECTURE.md) | Source-level: `_SymmetricMemory`, `empty_strided_p2p`, `rendezvous`, dispatch trees, backends |
| Implement async TP, Ulysses SP, custom P2P | [COMMON-PATTERNS.md](COMMON-PATTERNS.md) | Recipes: fused ops, workspace patterns, FP8, CUDA-graph capture, DO/DON'T pairs |
| Debug a hang in distributed training | Load `distributed-hang-diagnosis` | NCCL hang patterns, watchdog, flight recorder |

## Key APIs

### Allocation & Setup

| API | Location | Purpose |
|---|---|---|
| `symm_mem.empty(*size, dtype, device)` | `_symmetric_memory/__init__.py:2098` | Allocate symmetric tensor (CUDA/NVSHMEM backed) |
| `symm_mem.rendezvous(tensor, group)` | `_symmetric_memory/__init__.py:2156` | Collective rendezvous — returns `_SymmetricMemory` handle |
| `get_symm_mem_workspace(group_name, min_size)` | `_symmetric_memory/__init__.py:97` | Managed workspace; auto-expands if too small |
| `symm_mem.set_backend(name)` | `_symmetric_memory/__init__.py:2194` | Choose `"CUDA"`, `"NCCL"`, or `"NVSHMEM"` |
| `symm_mem.set_signal_pad_size(size)` | `_symmetric_memory/__init__.py:2230` | Must be called before any allocations |
| `symm_mem.is_nvshmem_available()` | `_symmetric_memory/__init__.py:2176` | Check NVSHMEM/rocSHMEM runtime availability |

### Fused Collective Ops

| Op | Signature | Use Case |
|---|---|---|
| `fused_all_gather_matmul` | `(A, Bs, gather_dim, group_name, return_A=True)` | Async TP forward (column parallel) |
| `fused_matmul_reduce_scatter` | `(A, B, reduce_op, scatter_dim, group_name)` | Async TP forward (row parallel) |
| `fused_all_gather_scaled_matmul` | `(A, Bs, A_scale, B_scales, ...)` | FP8 async TP |
| `fused_scaled_matmul_reduce_scatter` | `(A, B, A_scale, B_scale, ...)` | FP8 reduce-scatter |
| `_low_contention_all_gather` | `(tensor, group_name)` | SM-free gather when input is already in symm mem |
| `_low_contention_reduce_scatter` | `(tensor, reduce_op, group_name)` | SM-free scatter |
| `_low_contention_all_gather_ce_multicast` | `(tensor, group_name)` | NVSwitch copy-engine multicast gather |
| `_low_contention_all_gather_ce_multicast_out` | `(tensor, group_name, out)` | CE multicast with preallocated out (CUDA graph safe) |
| `get_remote_tensors` | `(x, group_name)` | Direct P2P tensor views — all ranks |
| `all_to_all_nd` | `(input, group_name, ...)` | Ulysses-style sequence parallel |

### Handle Methods (returned by `rendezvous`)

| Method | Purpose |
|---|---|
| `.rank` | Rank within the group |
| `.world_size` | Group size |
| `.barrier(channel=0)` | GPU-side barrier (spins on signal pad) |
| `.get_buffer(rank, shape, dtype)` | View into remote rank's symm mem buffer |
| `.get_remote_tensor(peer, size, dtype)` | Remote tensor slice |

## Fused Op Dispatch Decision Tree

```
fused_all_gather_matmul(A_shard, Bs, gather_dim, group_name)
        |
        v
  TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP set
  AND cuda AND contiguous AND gather_dim==0
  AND 2048 < local_M * world_size <= 4096
  AND single B?
        |
       YES → _fused_all_gather_matmul_native  (line 944)
        |       _async_input_mm; copy-pull on backend_stream
        |       uses stream_write_value32 for signal
       NO
        |
        v
  has_multicast_support (NVSwitch/NVLS)
  AND not return_A AND contiguous AND gather_dim==0
  AND local_M * world_size <= 2048?
        |
       YES → _multimem_all_gather_matmul  (line 1024)
        |       workspace multimem write; single GPU-kernel AG+MM
       NO
        |
        v
  _fused_all_gather_matmul_impl  (line 559)
        |
        ├── gather_dim == last dim?
        │     YES → _fused_all_gather_matmul_last_gather_dim_impl (line 767)
        │     NO  → pipelined backend_stream chunks
        |
        v
  Meta/CPU → _fused_all_gather_matmul_fallback  (line 836)
               (plain all_gather then matmul; no overlap)
```

## Common Failure Modes

| Failure | Cause | Fix |
|---|---|---|
| `RuntimeError: workspace size ... during graph capture` | `get_symm_mem_workspace` called inside CUDA graph with insufficient pre-allocated size | Call `get_symm_mem_workspace(group, min_size)` before starting capture |
| `RuntimeError: low_contention all-gather requires signal_pad_size >= N` | `world_size` too large for default signal pad; CE multicast needs channel 3 | Call `set_signal_pad_size(N)` before any allocations; `N = 4 * 4 * world_size` minimum |
| `ValueError: Tensor is not allocated from Symmetric Memory` | `get_remote_tensors` called on a tensor not created via `symm_mem.empty` | Allocate with `symm_mem.empty` or use `get_symm_mem_workspace` |
| `fused_all_gather_matmul` falls through to fallback silently | Tensor not contiguous, or `gather_dim != 0`, or FP8 scale shape wrong | Check contiguity; use `restride_A_shard_for_fused_all_gather_matmul` helper |
| Hang at `symm_mem.barrier()` | Ranks enter barrier at different points (asymmetric control flow) | All ranks must call the same sequence of barriers; verify no conditional barriers |
| `ImportError: _is_nvshmem_available` | NVSHMEM not compiled into this build | Check with `is_nvshmem_available()`; fall back to `"CUDA"` backend |
| `RuntimeError: {op} requires multicast support` | CE multicast op called on hardware without NVSwitch/NVLS | Gate with `_SymmetricMemory.has_multicast_support(DeviceType.CUDA, device.index)` |

## Debug Environment Variables

| Variable | Effect |
|---|---|
| `TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP=1` | Enable `_async_input_mm` native path in `fused_all_gather_matmul` |
| `TORCH_SYMMMEM_IMPLICIT_POOL=0` | Disable implicit memory pool for `symm_mem.empty` (use explicit `empty_strided_p2p` path) |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | Log process group creation and collective sequence |

## Quick Reference

```python
import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import (
    empty, rendezvous, get_symm_mem_workspace,
    set_backend, set_signal_pad_size,
)
import torch.ops.symm_mem as symm_mem_ops

# --- Allocation pattern ---
# allocate before CUDA graph capture
tensor = empty(numel, dtype=torch.float16, device="cuda")
hdl = rendezvous(tensor, group=pg)          # collective

# --- Workspace pattern ---
symm_mem = get_symm_mem_workspace(group_name, min_size=1 << 20)  # 1MB
buf = symm_mem.get_buffer(symm_mem.rank, shape, dtype)

# --- Fused async TP ---
A_full, [out] = torch.ops.symm_mem.fused_all_gather_matmul(
    A_shard, [B], gather_dim=0, group_name=pg.group_name
)

# --- Multicast gate ---
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory
if _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device.index):
    out = torch.ops.symm_mem._low_contention_all_gather_ce_multicast(tensor, group_name)
else:
    out = torch.ops.symm_mem._low_contention_all_gather(tensor, group_name)

# --- Signal pad sizing ---
set_signal_pad_size(4 * 4 * world_size)  # channels * sizeof(uint32) * world_size
# then allocate...
```

---

**Reference**: See [ARCHITECTURE.md](ARCHITECTURE.md) for C++ binding internals and dispatch implementation details.
**Reference**: See [COMMON-PATTERNS.md](COMMON-PATTERNS.md) for complete async TP, Ulysses SP, and CUDA graph capture recipes.
