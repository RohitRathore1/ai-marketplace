# Symmetric Memory Architecture

Source-level deep-dive into PyTorch's symmetric memory substrate — the P2P GPU communication layer behind async tensor parallelism.

**For usage patterns and recipes**: See [COMMON-PATTERNS.md](COMMON-PATTERNS.md)
**For API quick reference and failure modes**: See [SKILL.md](SKILL.md)

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [C++ Binding: `_SymmetricMemory`](#c-binding-_symmetricmemory)
4. [Allocator Lifecycle](#allocator-lifecycle)
5. [Signal Pad Layout](#signal-pad-layout)
6. [Python User-Facing API](#python-user-facing-api)
7. [Workspace Management](#workspace-management)
8. [Library Operator Registrations](#library-operator-registrations)
9. [Fused Op Dispatch Trees](#fused-op-dispatch-trees)
10. [Low-Contention Collective Variants](#low-contention-collective-variants)
11. [Backend Files](#backend-files)
12. [Key Files Reference](#key-files-reference)

---

## Overview

Symmetric memory maps the same physical GPU memory into all ranks within an NVLink domain simultaneously. Each rank can directly read from or write to another rank's buffer without a NCCL kernel launch or CPU round-trip. This enables two capabilities:

- **Fused async TP ops**: overlap all-gather/reduce-scatter with the matrix multiplication on a separate CUDA stream, hiding communication latency behind compute
- **Low-contention collectives**: perform collectives without using SMs — the copy engine or peer CUDA memcpy handles the data movement, leaving SMs free for compute

**Design philosophy**:
- Avoid NCCL kernel launches for short, latency-sensitive collectives
- Pipeline communication on a dedicated backend stream, not the compute stream
- Use signal pads (P2P-accessible uint32 arrays) for lightweight GPU-side barrier synchronization
- Support three allocation backends: CUDA P2P (default), NCCL, NVSHMEM

**Location**: `torch/distributed/_symmetric_memory/`

---

## Directory Structure

```
torch/distributed/_symmetric_memory/
├── __init__.py          # 2593 lines: Python API, fused ops, workspace management
├── _nccl.py             # 111 lines: external NCCL comm registration (NcclCommRegistration)
├── _nvshmem_triton.py   # 1220 lines: NVSHMEM + Triton kernel implementations
├── _rocshmem_triton.py  # 459 lines: ROCshmem + Triton kernel implementations (ROCm)
├── _shmem_triton.py     # 37 lines: shared-memory Triton utilities
└── _shmem_triton_utils.py # 101 lines: common Triton kernel helpers
```

**C++ implementation**: `torch/csrc/distributed/c10d/SymmetricMemory.hpp`, `SymmetricMemory.cpp` (allocator, rendezvous, signal pad, multicast detection)

**Python bindings**: `torch._C._distributed_c10d._SymmetricMemory` (the class imported at the top of `__init__.py`)

---

## C++ Binding: `_SymmetricMemory`

The `_SymmetricMemory` class is the primary C++ object exposed to Python. It represents an established symmetric memory allocation after rendezvous completes.

### Static Methods (class-level operations)

| Method | Signature | Purpose |
|---|---|---|
| `empty_strided_p2p` | `(size, stride, dtype, device, group_name=None)` | Allocate a tensor in P2P-accessible memory; no rendezvous yet |
| `rendezvous` | `(tensor, group_name)` | Collective: exchange buffer pointers and return handle; returns `None` if tensor not in symm mem |
| `set_group_info` | `(group_name, rank, world_size, store)` | Deprecated — was needed before lazy group registration |
| `has_multicast_support` | `(DeviceType, device_index)` | Check NVSwitch/NVLS hardware multicast capability |
| `stream_write_value32` | `(tensor, offset, val)` | Write `val` to `tensor[offset]` with system-level fence (non-graph) |
| `memset32` | `(tensor, offset, val, count)` | Bulk write for use inside CUDA graph capture |
| `signal_pad_size` | property (get/set) | Size in bytes of each allocation's signal pad region |
| `set_backend` | `(name: str)` | Set global allocation backend: `"CUDA"`, `"NCCL"`, `"NVSHMEM"` |
| `get_backend` | `(device: torch.device)` | Get current backend for a device |
| `get_mempool_allocator` | `(device)` | Get the MemPool allocator object for custom allocation |

### Instance Methods (on a rendezvous'd handle)

| Method / Attribute | Purpose |
|---|---|
| `.rank` | This process's rank in the symmetric memory group |
| `.world_size` | Number of processes in the group |
| `.barrier(channel=0)` | GPU-side barrier; spins on signal pad channel using atomic CAS |
| `.get_buffer(rank, shape, dtype)` | Zero-copy view into `rank`'s mapped buffer region |
| `.get_remote_tensor(peer, size, dtype)` | Slice view into a peer's buffer |

### How `rendezvous` Works

```
All ranks call _SymmetricMemory.rendezvous(tensor, group_name)
        |
        v
C++ layer: exchange buffer virtual addresses via the group's store
        |
        v
Each rank now holds a mapping: rank -> (ptr, size) for all peers
        |
        v
Returns _SymmetricMemory handle with .get_buffer() providing zero-copy views
```

If the tensor was not allocated via `empty_strided_p2p` (not in the P2P allocator pool), `rendezvous` returns `None`. Callers check for this and fall back to `get_symm_mem_workspace`.

---

## Allocator Lifecycle

```
1. empty() / empty_strided_p2p()
     Allocates memory in the P2P-capable allocator pool.
     Memory is NOT yet visible to peers. Returns a plain torch.Tensor.
     |
     v
2. rendezvous(tensor, group)  -- collective
     All ranks in the group call this with their local tensor.
     C++ layer exchanges buffer addresses via the process group's store.
     Returns _SymmetricMemory handle; tensor is now peer-accessible.
     |
     v
3. get_buffer(rank, shape, dtype)
     Zero-copy view into another rank's allocated region.
     Reads/writes go directly over NVLink, no kernel launch.
     |
     v
4. barrier(channel=0)
     GPU-side spin barrier using signal pad slots.
     Must be called symmetrically on all ranks.
     |
     v
5. Tensor lifetime ends → allocator reclaims P2P memory
```

**Critical constraint**: All ranks must call `rendezvous` with tensors of **identical shape, dtype, and device type**. A mismatch causes a hang inside the C++ store exchange.

**CUDA graph capture constraint**: `get_symm_mem_workspace` cannot expand during capture. Call it with the required `min_size` before initiating capture.

---

## Signal Pad Layout

Signal pads are small P2P-accessible `uint32` arrays stored at the **front** of each symmetric memory allocation. They are used for GPU-side synchronization via atomic compare-and-swap.

```
Symmetric memory allocation layout (post PR #189088):
┌──────────────────────────────────────────────────────────┐
│  Signal Pad  (signal_pad_size bytes, at offset 0)        │
│  ┌──────────┬──────────┬──────────┬──────────────────┐   │
│  │ channel0 │ channel1 │ channel2 │ channel3 (CE MC) │   │
│  │ W ranks  │ W ranks  │ W ranks  │ W ranks          │   │
│  │ uint32[] │ uint32[] │ uint32[] │ uint32[]         │   │
│  └──────────┴──────────┴──────────┴──────────────────┘   │
│                                                          │
│  Data Region  (remainder of allocation)                  │
└──────────────────────────────────────────────────────────┘
```

- Each channel occupies `world_size * sizeof(uint32)` = `world_size * 4` bytes
- `barrier(channel=N)` atomically sets this rank's slot to 1, then spin-waits until all other ranks' slots become 1, then resets to 0
- `_CE_MULTICAST_BARRIER_CHANNEL = 3` — the copy-engine multicast ops use a dedicated channel to avoid interfering with the standard barrier channels
- **Minimum signal pad size for CE multicast**: `(3+1) * world_size * 4` bytes
- `set_signal_pad_size(size)` must be called before the first allocation

**Why signal pads moved to the front** (PR #189088): placing them at offset 0 ensures the signal pad's physical address is always at the start of the P2P mapping, avoiding potential alignment or offset computation errors in kernels that compute signal pad addresses from the base pointer.

---

## Python User-Facing API

All user-facing functions are in `__init__.py` below the `# User-facing APIs` comment at line 2050.

### `empty(*size, dtype, device)` — line 2098

Allocates a symmetric memory tensor. By default uses the implicit memory pool (`TORCH_SYMMMEM_IMPLICIT_POOL=1`), which wraps `empty_strided_p2p` inside `torch.cuda.use_mem_pool(mempool)`. Set the env var to `0` to bypass the pool and call `empty_strided_p2p` directly.

### `rendezvous(tensor, group)` — line 2156

Thin Python wrapper around `_SymmetricMemory.rendezvous(tensor, group_name)`. Accepts either a `ProcessGroup` object or a group name string. Returns the `_SymmetricMemory` handle.

### `set_backend(name)` / `get_backend(device)` — lines 2194, 2208

Switch the global allocation backend. The backend cannot be changed after any allocation has been performed. Use cases:
- `"CUDA"`: default, uses CUDA P2P via NVLink
- `"NCCL"`: uses NCCL as the transport (useful for non-NVLink topologies)
- `"NVSHMEM"`: uses NVSHMEM; required for NVSHMEM Triton kernels

### `set_signal_pad_size(size)` / `get_signal_pad_size()` — lines 2230, 2255

Configure the signal pad before any allocations. The default size is hardware-dependent. Required to be increased when `world_size` is large or when using the CE multicast ops with many ranks.

### `is_nvshmem_available()` — line 2176

Checks `_is_nvshmem_available` from the C extension. On ROCm, also verifies rocSHMEM version >= 3.3.0.

### Deprecated: `enable_symm_mem_for_group` — line 30

No longer needed. Previously called `_SymmetricMemory.set_group_info` to register a store for the group. Group registration is now automatic on first rendezvous.

---

## Workspace Management

`get_symm_mem_workspace(group_name, min_size)` — line 97

The workspace is a per-group reusable `uint8` scratch buffer in symmetric memory. Fused ops that need scratch space call this rather than managing their own allocation.

```python
def get_symm_mem_workspace(group_name, min_size):
    tensor = _group_name_to_workspace_tensor.get(group_name)
    size = tensor.numel() * tensor.element_size() if tensor else 0
    if tensor is None or size < min_size:
        # Raises if inside CUDA graph capture (can't expand during capture)
        tensor = _SymmetricMemory.empty_strided_p2p(
            (max(size, min_size),), [1], torch.uint8, device, group_name
        )
        _group_name_to_workspace_tensor[group_name] = tensor
    return _SymmetricMemory.rendezvous(tensor)
```

**Important behaviors:**
- Workspace is monotonically growing — once expanded, never shrinks
- Each `rendezvous` call inside returns a fresh handle (re-exchange of pointers)
- Calling with `min_size` larger than current during CUDA graph capture raises `RuntimeError` — callers must pre-warm before capture begins

---

## Library Operator Registrations

All fused ops are registered under the `symm_mem` custom operator namespace:

```python
lib = torch.library.Library("symm_mem", "DEF")
```

Access via `torch.ops.symm_mem.<op_name>`.

All definitions include `tags=[torch._C.Tag.needs_fixed_stride_order]` — this prevents inductor from reordering strides in ways that would break the P2P memory layout assumptions.

Each op has three implementations registered:
- `"Meta"` — shape inference only, no actual communication
- `"CUDA"` — production implementation with dispatch logic
- `"XPU"` — same as CUDA for Intel XPU support (where applicable)

---

## Fused Op Dispatch Trees

### `fused_all_gather_matmul` — lines 868-916

```
fused_all_gather_matmul(A_shard, Bs, gather_dim, group_name, return_A)
        |
        ├── _should_use_fused_all_gather_matmul_native (line 919)
        │   Conditions: TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP
        │               cuda, contiguous, gather_dim==0
        │               2048 < local_M * world_size <= 4096
        │               single B, local_M % world_size == 0
        │   Path: _fused_all_gather_matmul_native (line 944)
        │         Uses _async_input_mm; fetches remote shards on backend_stream
        │         stream_write_value32 (or memset32 for graph) signals each shard ready
        │
        ├── _should_use_multimem_all_gather_matmul (line 998)
        │   Conditions: has_multicast_support (NVSwitch/NVLS)
        │               not return_A, contiguous, gather_dim==0
        │               local_M * world_size <= 2048
        │   Path: _multimem_all_gather_matmul (line 1024)
        │         Workspace write via multimem; single fused kernel
        │
        └── _fused_all_gather_matmul_impl (line 559)
            │   Pipelined: per-chunk copy on backend_stream interleaved with matmul
            │
            ├── gather_dim == last dim? → _fused_all_gather_matmul_last_gather_dim_impl (line 767)
            └── else → _pipelined_all_gather_and_consume (line 300)
```

### `fused_matmul_reduce_scatter` — lines 1225-1359

```
fused_matmul_reduce_scatter(A, B, reduce_op, scatter_dim, group_name)
        |
        v
_fused_matmul_reduce_scatter_impl (line 1273)
        |
        ├── Compute output chunks on compute stream
        ├── For each chunk: async reduce-scatter via symm_mem on backend_stream
        └── Final barrier; output is scattered result

Fallback (Meta/CPU): _fused_matmul_reduce_scatter_fallback (line 1261)
        plain matmul then reduce_scatter; no overlap
```

### `_low_contention_all_gather` dispatch — lines 1705-1749

```
_low_contention_all_gather(tensor, group_name)
        |
        ├── rendezvous(tensor, group_name) → not None?
        │   YES: input_is_symm_mem = True (no extra copy needed)
        │   NO:  get_symm_mem_workspace; copy tensor into local buffer
        |
        └── On backend_stream:
              barrier(channel=0)
              for each step: chunks[remote_rank].copy_(symm_mem.get_buffer(...))
              barrier(channel=0)
              _register_work(output)  # work token for completion tracking
```

---

## Low-Contention Collective Variants

These variants avoid NCCL kernel launches entirely:

| Variant | SM Usage | NVSwitch Required | CUDA Graph Safe |
|---|---|---|---|
| `_low_contention_all_gather` | Minimal (copy only) | No | Yes (with workspace pre-warm) |
| `_low_contention_reduce_scatter` | Minimal (atomics) | No | Yes |
| `_low_contention_all_gather_ce_multicast` | None (copy engine) | Yes (NVSwitch) | No (allocates output) |
| `_low_contention_all_gather_ce_multicast_out` | None (copy engine) | Yes | Yes (preallocated out) |

**CE multicast barrier channel**: `_CE_MULTICAST_BARRIER_CHANNEL = 3` (line 1761). The two barriers in the CE multicast path (pre-write and post-write) are ordered on the same stream — if a peer has not consumed the first signal yet, the second barrier's CAS spins until that slot is cleared.

---

## Backend Files

### `_nccl.py` — External NCCL Comm Registration

Provides `register_external_nccl_comm(group_name, comm_ptr, device, comm=None)` for non-`ProcessGroupNCCL` backends (e.g., torchcomms) to publish an existing `ncclComm_t` into PyTorch's symmetric memory registry. Returns a `NcclCommRegistration` context manager that unregisters on exit or garbage collection.

**Why this exists**: `_SymmetricMemory.rendezvous` needs access to the group's NCCL communicator to complete the pointer exchange. For backends that own the NCCL communicator externally (without going through `ProcessGroupNCCL`), they must register it first.

### `_nvshmem_triton.py` — NVSHMEM + Triton Kernels (1220 lines)

Contains Triton kernel implementations for:
- All-gather using NVSHMEM `nvshmem_getmem` calls
- Reduce-scatter using NVSHMEM `nvshmem_putmem` + atomic reduction
- `all_to_all_nd` for Ulysses-style sequence parallel

These are activated when `set_backend("NVSHMEM")` is called. NVSHMEM provides a symmetric heap backed by NVSHMEM's `nvshmem_malloc`.

### `_rocshmem_triton.py` — ROCshmem + Triton Kernels (459 lines)

ROCm equivalent of `_nvshmem_triton.py`. Activated automatically on ROCm builds when `is_nvshmem_available()` returns True (which checks rocSHMEM >= 3.3.0).

### `_shmem_triton.py` / `_shmem_triton_utils.py` — Shared Triton Utilities

Common helpers used by both NVSHMEM and ROCshmem backends: signal pad access patterns, barrier implementations in Triton, buffer address computation.

---

## Key Files Reference

| File | Lines | Content |
|---|---|---|
| `torch/distributed/_symmetric_memory/__init__.py` | 2593 | All Python API, fused ops, dispatch logic |
| `torch/distributed/_symmetric_memory/_nccl.py` | 111 | External NCCL comm registration |
| `torch/distributed/_symmetric_memory/_nvshmem_triton.py` | 1220 | NVSHMEM + Triton kernels |
| `torch/distributed/_symmetric_memory/_rocshmem_triton.py` | 459 | ROCshmem + Triton kernels |
| `torch/csrc/distributed/c10d/SymmetricMemory.hpp` | — | C++ `_SymmetricMemory` class definition |
| `torch/csrc/distributed/c10d/SymmetricMemory.cpp` | — | Allocator, rendezvous, signal pad, multicast detection |
| `torch/_C/_distributed_c10d.pyi.in` | — | Python stubs for `_SymmetricMemory` class |

### Key Function Map

| Function | File:Line | Purpose |
|---|---|---|
| `get_symm_mem_workspace` | `__init__.py:97` | Managed scratch workspace with auto-grow |
| `_pipelined_all_gather_and_consume` | `__init__.py:300` | Core pipelined AG+consumer implementation |
| `_fused_all_gather_matmul_impl` | `__init__.py:559` | Main dispatch for fused AG+MM |
| `_should_use_fused_all_gather_matmul_native` | `__init__.py:919` | Native path gate check |
| `_fused_all_gather_matmul_native` | `__init__.py:944` | `_async_input_mm` + stream_write_value32 path |
| `_should_use_multimem_all_gather_matmul` | `__init__.py:998` | Multimem path gate check |
| `_low_contention_all_gather` | `__init__.py:1705` | SM-free all-gather implementation |
| `_check_lc_signal_pad_capacity` | `__init__.py:1794` | Validate signal pad is large enough |
| `_low_contention_all_gather_ce_multicast_impl` | `__init__.py:1869` | CE multicast gather implementation |
| `register_external_nccl_comm` | `_nccl.py:70` | Register external NCCL comm for rendezvous |
