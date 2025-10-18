import sys
sys.path.append('build/')
import torch
import time
import mixedgemm  
for i in range(1):
    M, N, K = 128, 4096, 4096
    group = 32
    KN, KS, KO = 4096 - 1024, 0, 1024

    torch.manual_seed(721)

    signs = (torch.randint(0, 2, (M, K), device='cuda', dtype=torch.bfloat16) * 2 - 1)
    X = torch.rand(M, K, dtype=torch.bfloat16, device='cuda') * 3
    if KN != 0:
        X[:, -KN:] = torch.rand(M, KN, dtype=torch.bfloat16, device='cuda') * 8 + 8
    if KS != 0:
        X[:, -KS:] = torch.rand(M, KS, dtype=torch.bfloat16, device='cuda') * 16 + 16
    if KO != 0:
        X[:, -KO:] = torch.rand(M, KO, dtype=torch.bfloat16, device='cuda') * 32 + 32
    X = X * signs

    W = torch.rand(N, K, dtype=torch.bfloat16, device='cuda') * 1

    reorder_index = torch.arange(K, dtype=torch.int16, device='cuda') 

    # KN, KS, KO = 4096, 0, 0

    WT = W.t().clone()
    AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(X, reorder_index, KN, KS, KO)
    
    # BN, BS, BO, SFBN, SFBS, SFBO = mixedgemm.reorder_quantize_w(W, reorder_index, KN, KS, KO)
    BN, BS, BO, SFBN, SFBS, SFBO = mixedgemm.reorder_quantize_w4(W, reorder_index, KN, KS, KO)


    C = mixedgemm.matmul(AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO)

    D = torch.matmul(X.to(torch.float32), WT.to(torch.float32))


    mean_value = torch.mean(C)

    variance_value = torch.var(C)

    mean_valued = torch.mean(D)

    variance_valued = torch.var(D)

    # E = C - D
    mse_loss_fn = torch.nn.MSELoss()
    mse_alt = mse_loss_fn(C, D)
    # variance_error = torch.var(E)

    print(f"平均值c: {mean_value.item():.6f}")
    print(f"方差c: {variance_value.item():.6f}")
    print(f"平均值d: {mean_valued.item():.6f}")
    print(f"方差d: {variance_valued.item():.6f}")
    print(f"valueC:{C.flatten()[:10]}...{C.flatten()[-10:]}")
    print(f"valueD:{D.flatten()[:10]}...{D.flatten()[-10:]}")
    print(f"误差E: {mse_alt.item()}")
    print(f"finish {i}")
    time.sleep(1)