"""Cold-L2, CUDA-graph timing core for the SM121 bf16 GEMV task.

Measurement contract (nvidia_agent_tasks/docs/measurement_contract.md):
  * cold L2 by default -- via R rotating weight copies so no weight stays
    resident (R chosen so R*weight_bytes >= 2*L2). Avoids subtracting a large
    flush from a small kernel.
  * CUDA-graph replay is the timer of record: production runs these ops inside
    full_cuda_graph_backend, so eager per-launch overhead must not be counted.
  * interleaved A/B arms; median over trials.

For weights larger than 2*L2 the single copy is already cold on every call.
"""

from __future__ import annotations

import math
import statistics

import torch

_L2 = None


def l2_bytes(device) -> int:
    global _L2
    if _L2 is None:
        _L2 = int(torch.cuda.get_device_properties(device).L2_cache_size)
    return _L2


def weight_copies(weight_bytes: int, device, copies_cap: int = 256) -> int:
    """Rotating copies so each timed call reads an L2-cold weight."""
    need = 2 * l2_bytes(device)
    r = math.ceil(need / max(weight_bytes, 1))
    return max(1, min(copies_cap, r))


class GraphBench:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)

    def build(self, fn, reps: int, cold: bool) -> torch.cuda.CUDAGraph:
        """Graph of `reps` calls. cold=True rotates buffers (distinct i);
        cold=False reuses buffer 0 (L2-warm)."""
        g = torch.cuda.CUDAGraph()
        if cold:
            ops = [(lambda i=i: fn(i)) for i in range(reps)]
        else:
            ops = [(lambda: fn(0)) for _ in range(reps)]
        with torch.cuda.graph(g):
            for op in ops:
                op()
        return g

    @staticmethod
    def measure(g, reps: int, replay: int = 3) -> float:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(replay):
            g.replay()
        e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1000.0 / (replay * reps)


def bandwidth_gbps(total_bytes: int, us: float) -> float:
    if us <= 0:
        return float("nan")
    return total_bytes / (us * 1e-6) / 1e9


def median(values):
    return statistics.median(values)
