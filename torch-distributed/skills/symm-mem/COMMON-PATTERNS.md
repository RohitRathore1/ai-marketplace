# Symmetric Memory Common Patterns

Implementation recipes for async tensor parallelism, sequence parallelism, low-contention collectives, FP8 fused ops, and CUDA graph capture with symmetric memory.

**For API reference and failure modes**: See [SKILL.md](SKILL.md)
**For C++ internals and dispatch trees**: See [ARCHITECTURE.md](ARCHITECTURE.md)

## Skill Navigation

| Your Goal | Use This File | Use `distributed-hang-diagnosis` |
|---|---|---|
| Implement async TP or sequence parallel | Yes — see Patterns 1–4 | No |
| Debug a symm_mem hang or deadlock | Yes — see Debugging section | Yes — if NCCL-level, not symm_mem |
| Add multicast collective | Yes — Pattern 5 | No |
| Understand why fused op uses fallback | Yes — Pattern 7 / dispatch section | No |
| Training hangs with NCCL timeout | No | Yes |

---

## Pattern 1: Async TP — Column Parallel (fused_all_gather_matmul)

Forward pass of a column-parallel linear layer. All-gather the input shard while running matmul on ready chunks, overlapping comms and compute.

```python
import torch
import torch.distributed as dist
import torch.ops.symm_mem

def column_parallel_forward(A_shard: torch.Tensor, B: torch.Tensor, pg: dist.ProcessGroup):
    """
    A_shard: local input shard [local_M, K]
    B:       weight matrix [K, N] (replicated)
    Returns: (A_full [M, K], out [M, N])
    """
    # A_shard must be contiguous for the pipelined path
    A_shard = A_shard.contiguous()

    A_full, [out] = torch.ops.symm_mem.fused_all_gather_matmul(
        A_shard,
        [B],
        gather_dim=0,
        group_name=pg.group_name,
        return_A=True,    # set False to skip returning A_full if not needed
    )
    return A_full, out
```

**Reshape helper**: If `A_shard` is not a clean 2D shard along dim 0, use the provided utility:

```python
from torch.distributed._symmetric_memory import restride_A_shard_for_fused_all_gather_matmul
A_shard = restride_A_shard_for_fused_all_gather_matmul(A_shard, gather_dim=0)
```

**When it falls to pipelined impl (most common case)**:
- `local_M * world_size > 4096` (native path threshold)
- `return_A=True` (multimem path skipped when A is needed)
- Non-CUDA device

**When it uses multimem (NVSwitch-only, small M)**:
- `has_multicast_support` + `local_M * world_size <= 2048` + `return_A=False`

---

## Pattern 2: Async TP — Row Parallel (fused_matmul_reduce_scatter)

Backward pass of a column-parallel linear or forward pass of a row-parallel linear. Runs matmul chunks and scatters partial sums without waiting for the full matmul to complete.

```python
def row_parallel_forward(A: torch.Tensor, B_shard: torch.Tensor, pg: dist.ProcessGroup):
    """
    A:       full input [M, K]
    B_shard: local weight shard [K, local_N]
    Returns: scattered output [M, local_N]
    """
    out = torch.ops.symm_mem.fused_matmul_reduce_scatter(
        A,
        B_shard,
        reduce_op="avg",   # or "sum"
        scatter_dim=0,
        group_name=pg.group_name,
    )
    return out
```

**Reshape helper**:

```python
from torch.distributed._symmetric_memory import restride_A_for_fused_matmul_reduce_scatter
A = restride_A_for_fused_matmul_reduce_scatter(A, scatter_dim=0)
```

---

## Pattern 3: Full Async TP Loop

Both sides pipelined together — the standard pattern for a tensor-parallel transformer layer.

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.ops.symm_mem


class AsyncTPLinear(nn.Module):
    def __init__(self, in_features, out_features, pg: dist.ProcessGroup):
        super().__init__()
        self.pg = pg
        world_size = pg.size()
        # shard weight column-wise
        self.weight_col = nn.Parameter(
            torch.empty(in_features, out_features // world_size, device="cuda")
        )

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        # x_shard: [local_M, in_features]
        x_shard = x_shard.contiguous()

        # AG + MM: gather input shards, compute full column output
        _, [partial] = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard,
            [self.weight_col],
            gather_dim=0,
            group_name=self.pg.group_name,
            return_A=False,
        )
        # partial: [full_M, local_out_features]
        return partial


class AsyncTPLinearRS(nn.Module):
    """Row parallel with reduce-scatter."""

    def __init__(self, in_features, out_features, pg: dist.ProcessGroup):
        super().__init__()
        self.pg = pg
        world_size = pg.size()
        self.weight_row = nn.Parameter(
            torch.empty(in_features // world_size, out_features, device="cuda")
        )

    def forward(self, x_full: torch.Tensor) -> torch.Tensor:
        # x_full: [M, in_features]
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x_full,
            self.weight_row,
            reduce_op="sum",
            scatter_dim=0,
            group_name=self.pg.group_name,
        )
```

---

## Pattern 4: Ulysses Sequence Parallel (all_to_all_nd)

Ulysses-style sequence parallelism exchanges sequence chunks so each rank processes a full sequence slice for its attention heads.

```python
import torch.ops.symm_mem


def ulysses_all_to_all(x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    """
    Input:  [seq_len // world_size, num_heads, head_dim]
    Output: [seq_len, num_heads // world_size, head_dim]
    """
    return torch.ops.symm_mem.all_to_all_nd(
        x,
        group_name=pg.group_name,
    )
```

**For custom split/merge dimensions** use the `in_splits` / `out_splits` arguments of `all_to_all_nd` (see `lib.define` in `__init__.py` for the `all_to_all_vdev_2d` family).

---

## Pattern 5: Low-Contention All-Gather

When the input is already in symmetric memory (e.g., the layer's weight is pre-allocated there), no SM copy is needed — the copy engine pulls directly from peers.

```python
from torch.distributed._symmetric_memory import empty, rendezvous
import torch.ops.symm_mem


def preallocate_symm_weight(shape, dtype, pg: dist.ProcessGroup):
    """Allocate weight in symmetric memory and rendezvous."""
    w = empty(*shape, dtype=dtype, device="cuda")
    w.copy_(torch.randn(*shape, dtype=dtype))   # initialize locally
    rendezvous(w, pg)                            # collective; peers can now access
    return w


def low_contention_gather(weight_shard: torch.Tensor, pg: dist.ProcessGroup):
    """If weight_shard is already in symm mem, no extra copy is needed."""
    return torch.ops.symm_mem._low_contention_all_gather(
        weight_shard,
        group_name=pg.group_name,
    )
```

**When input is NOT in symm mem**, `_low_contention_all_gather` falls back to:
1. Get workspace via `get_symm_mem_workspace`
2. Copy input into workspace on backend stream
3. Pull from all peers

---

## Pattern 6: Multicast All-Gather (NVSwitch / NVLS)

The copy-engine multicast path is zero-SM and CUDA-graph safe (with the `_out` variant).

```python
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import empty, rendezvous, get_signal_pad_size, set_signal_pad_size
import torch.ops.symm_mem


def setup_multicast(world_size: int):
    """Call before any allocations when using CE multicast."""
    # CE multicast uses channel 3; minimum pad = 4 channels * world_size * 4 bytes
    min_pad = 4 * world_size * 4
    if get_signal_pad_size() < min_pad:
        set_signal_pad_size(min_pad)


def multicast_all_gather(tensor: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    device_index = tensor.device.index
    if _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device_index):
        return torch.ops.symm_mem._low_contention_all_gather_ce_multicast(
            tensor, group_name=pg.group_name
        )
    # Fallback to standard low-contention gather
    return torch.ops.symm_mem._low_contention_all_gather(
        tensor, group_name=pg.group_name
    )


def multicast_all_gather_graph_safe(
    tensor: torch.Tensor, out: torch.Tensor, pg: dist.ProcessGroup
) -> torch.Tensor:
    """
    Pre-allocate `out` before CUDA graph capture, then use the _out variant.
    `out` must be the right shape (tensor.shape[0] * world_size, *tensor.shape[1:]).
    """
    return torch.ops.symm_mem._low_contention_all_gather_ce_multicast_out(
        tensor, group_name=pg.group_name, out=out
    )
```

---

## Pattern 7: FP8 Fused Ops

For FP8 (float8_e4m3fn) async TP with scaling.

```python
def fp8_column_parallel_forward(
    A_shard: torch.Tensor,       # float8_e4m3fn, [local_M, K]
    A_scale: torch.Tensor,       # float32, tensor-wise or row-wise
    B: torch.Tensor,             # float8_e4m3fn, [K, N]
    B_scale: torch.Tensor,       # float32, tensor-wise
    pg: dist.ProcessGroup,
):
    A_shard = A_shard.contiguous()
    _, [out] = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
        A_shard,
        [B],
        A_scale,
        [B_scale],
        gather_dim=0,
        group_name=pg.group_name,
        biases=[None],
        result_scales=[None],
        out_dtypes=[torch.float16],
        use_fast_accum=[True],
    )
    return out
```

**Scale shape rules** (from `_check_and_verify_fp8_all_gather_scale_mode`):
- Tensor-wise: `A_scale.numel() == 1`
- Row-wise sharded: `A_scale.shape[:-1] == A_shard.shape[:-1]` and `A_scale.shape[-1] == 1`
- Row-wise replicated: `A_scale.shape[:-1] == full_A.shape[:-1]` (gathered shape)

---

## Pattern 8: CUDA Graph Capture

Symmetric memory requires pre-warming the workspace and pre-allocating outputs before graph capture begins.

```python
from torch.distributed._symmetric_memory import get_symm_mem_workspace
import torch.ops.symm_mem


def prepare_for_cuda_graph(
    A_shard_shape, B_shape, group_name: str, world_size: int
):
    """Call this BEFORE torch.cuda.graph() context."""
    local_M = A_shard_shape[0]
    K = A_shard_shape[1]
    # Workspace needed for AG: local_M * K * element_size
    min_workspace = local_M * K * 2  # float16 = 2 bytes
    get_symm_mem_workspace(group_name, min_size=min_workspace)

    # For CE multicast _out: pre-allocate output
    out_shape = (A_shard_shape[0] * world_size, B_shape[1])
    out = torch.empty(out_shape, dtype=torch.float16, device="cuda")
    return out


def run_with_cuda_graph(A_shard, B, out_buf, pg):
    # Warmup
    for _ in range(3):
        torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, [B], gather_dim=0, group_name=pg.group_name
        )

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        result = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, [B], gather_dim=0, group_name=pg.group_name
        )
    return g, result
```

---

## Pattern 9: get_remote_tensors for Custom P2P Kernels

When writing a custom Triton kernel that reads from all peers simultaneously.

```python
import torch.ops.symm_mem
from torch.distributed._symmetric_memory import empty, rendezvous


def custom_p2p_kernel(local_tensor: torch.Tensor, pg: dist.ProcessGroup):
    """
    local_tensor must have been allocated via symm_mem.empty.
    Returns tuple of tensors: one view per rank, ordered by rank ID.
    Works only within NVLink domain (world_within_direct_access()).
    """
    remote_views = torch.ops.symm_mem.get_remote_tensors(
        local_tensor, group_name=pg.group_name
    )
    # remote_views[rank] is a zero-copy view into that rank's symm mem buffer
    # Pass to a Triton kernel that reads from multiple views in parallel
    return remote_views
```

---

## Anti-Patterns (Avoid These)

### Expanding workspace during CUDA graph capture

```python
# WRONG: workspace expansion inside graph raises RuntimeError
with torch.cuda.graph(g):
    symm_mem = get_symm_mem_workspace(group_name, min_size=too_large)
    ...

# CORRECT: pre-warm before capture
get_symm_mem_workspace(group_name, min_size=required_size)  # outside graph
with torch.cuda.graph(g):
    symm_mem = get_symm_mem_workspace(group_name, min_size=required_size)  # uses cached
    ...
```

### Calling rendezvous with mismatched tensor shapes

```python
# WRONG: different shapes across ranks cause hang inside C++ store exchange
if rank == 0:
    t = symm_mem.empty(1024, dtype=torch.float16, device="cuda")
else:
    t = symm_mem.empty(2048, dtype=torch.float16, device="cuda")
rendezvous(t, pg)  # hangs

# CORRECT: identical shape on all ranks
t = symm_mem.empty(1024, dtype=torch.float16, device="cuda")
rendezvous(t, pg)  # works
```

### Using get_remote_tensors on a non-symm-mem tensor

```python
# WRONG: raises ValueError
t = torch.randn(128, device="cuda")
views = torch.ops.symm_mem.get_remote_tensors(t, pg.group_name)  # ValueError

# CORRECT: allocate via symm_mem.empty
t = symm_mem.empty(128, dtype=torch.float32, device="cuda")
rendezvous(t, pg)
views = torch.ops.symm_mem.get_remote_tensors(t, pg.group_name)
```

### Calling barrier asymmetrically

```python
# WRONG: only rank 0 calls barrier — all other ranks hang
if rank == 0:
    hdl.barrier(channel=0)

# CORRECT: all ranks call barrier
hdl.barrier(channel=0)
```

### Changing backend after allocations

```python
# WRONG: raises RuntimeError — backend cannot change after first allocation
t = symm_mem.empty(128, device="cuda")
symm_mem.set_backend("NVSHMEM")   # error

# CORRECT: set backend before any allocations
symm_mem.set_backend("NVSHMEM")
t = symm_mem.empty(128, device="cuda")
```

### Skipping signal pad sizing for large world sizes with CE multicast

```python
# WRONG: default signal_pad_size may be too small for large world_size
# _check_lc_signal_pad_capacity raises RuntimeError silently
torch.ops.symm_mem._low_contention_all_gather_ce_multicast(t, group_name)

# CORRECT: set signal pad size before any allocations
world_size = pg.size()
set_signal_pad_size(4 * 4 * world_size)   # 4 channels * sizeof(uint32) * world_size
# then allocate
```

---

## Debugging Patterns

### Check which dispatch path fused_all_gather_matmul uses

```python
import os
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable native async TP path
os.environ["TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP"] = "1"

# Force fallback (Meta device registration fires for shape tracing)
A_shard_meta = A_shard.to("meta")
_ = torch.ops.symm_mem.fused_all_gather_matmul(
    A_shard_meta, [B.to("meta")], gather_dim=0, group_name=pg.group_name
)
```

### Verify multicast support on current hardware

```python
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory

for i in range(torch.cuda.device_count()):
    has_mc = _SymmetricMemory.has_multicast_support(DeviceType.CUDA, i)
    print(f"GPU {i}: multicast_support={has_mc}")
```

### Check if tensor is in symmetric memory

```python
from torch.distributed._symmetric_memory import rendezvous

hdl = rendezvous(tensor, pg)
if hdl is None:
    print("NOT in symmetric memory — will use workspace fallback")
else:
    print(f"In symmetric memory, rank={hdl.rank}, world_size={hdl.world_size}")
```

### Debug signal pad capacity

```python
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import get_symm_mem_workspace

symm_mem = get_symm_mem_workspace(group_name, min_size=1)
world_size = symm_mem.world_size
signal_pad_bytes = _SymmetricMemory.signal_pad_size
required = 4 * world_size * 4   # 4 channels * sizeof(uint32) * world_size
print(f"signal_pad_size={signal_pad_bytes}, required={required}, ok={signal_pad_bytes >= required}")
```

---

## Quick File Reference

| Task | File | What to Modify |
|---|---|---|
| Add new fused op | `__init__.py` | `lib.define(...)`, `@torch.library.impl(lib, ..., "CUDA")` |
| Add NVSHMEM Triton kernel | `_nvshmem_triton.py` | New `@triton.jit` kernel + registration |
| Register external NCCL comm | `_nccl.py` | Use `register_external_nccl_comm` API |
| Change signal pad default | `__init__.py:signal_pad_size` | `_SymmetricMemory.signal_pad_size = N` |
| Add ROCm backend kernel | `_rocshmem_triton.py` | Mirror the NVSHMEM pattern for rocSHMEM |
| Gate new op on multicast hw | `__init__.py` | `_SymmetricMemory.has_multicast_support(DeviceType.CUDA, idx)` |
| Debug allocator internals | `torch/csrc/distributed/c10d/SymmetricMemory.cpp` | Allocator, rendezvous, store exchange |
