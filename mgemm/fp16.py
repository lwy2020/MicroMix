import torch
from torch.nn import functional as F

test_M_pool = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
@torch.no_grad()
def test_quant_linear_a16_w16(M, N ,K) -> float:
    weight = torch.rand(K, N, dtype=torch.float16).cuda()
    x = torch.rand(M, K, dtype=torch.float16).cuda()

    elapsed_time_ms = 0
    iterations = 300

    for _ in range(iterations//10):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        y = F.linear(x, weight)
        end_event.record()
        torch.cuda.synchronize()
    
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        y = F.linear(x, weight)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms += start_event.elapsed_time(end_event)

    total_ops = M * N * K * 2 * iterations
    gflops = total_ops / elapsed_time_ms / 10**9
    return gflops, elapsed_time_ms / iterations

a16w16_ms = []
for m in test_M_pool:
    gflops, elapsed_time_ms = test_quant_linear_a16_w16(m, 4096, 4096)
    a16w16_ms.append(elapsed_time_ms)

print(a16w16_ms)