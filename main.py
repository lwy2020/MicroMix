import sys
sys.path.append('mgemm/build/')
import torch
import time
import mixedgemm  
for i in range(1):
    M, N, K = 128, 4096, 4096
    group = 32
    KN, KS, KO = 4096 - 2048, 2048-256, 256

    torch.manual_seed(721)
    # X = torch.ones(M, K, dtype=torch.bfloat16, device='cuda') * 1
    # X[0, 1] = -2.5
    # W = torch.ones(N, K, dtype=torch.bfloat16, device='cuda') * 0.5
    signs = (torch.randint(0, 2, (M, K), device='cuda', dtype=torch.bfloat16) * 2 - 1)
    X = torch.rand(M, K, dtype=torch.bfloat16, device='cuda') * 3
    X[:, -KS:] = torch.rand(M, KS, dtype=torch.bfloat16, device='cuda') * 8 + 8
    X[:, -KO:] = torch.rand(M, KO, dtype=torch.bfloat16, device='cuda') * 16 + 16
    X[:, -16:] = torch.rand(M, 16, dtype=torch.bfloat16, device='cuda') * 32 + 32
    X = X * signs
    # X[:, -KS:] = torch.full((M, KS), float('nan'), dtype=torch.bfloat16, device='cuda')
    # X[:, -KN:] = torch.full((M, KN), float('nan'), dtype=torch.bfloat16, device='cuda')
    # X = torch.randint(-3, 3, (M, K), dtype=torch.bfloat16, device='cuda')
    # X[:, -KS:] = torch.randint(-14, 14, (M, KS), dtype=torch.bfloat16, device='cuda') * 1
    # X[:, -KN:] = torch.randint(-16, 16, (M, KN), dtype=torch.bfloat16, device='cuda') * 2
    W = torch.rand(N, K, dtype=torch.bfloat16, device='cuda') * 2
    # torch.nn.init.kaiming_normal_(
    #     W,
    #     mode='fan_in',        # 'fan_in' 使前向传播时方差保持不变
    #     nonlinearity='relu'   # 专为 ReLU 激活函数设计
    # )
    # W = torch.randint(-3, 3, (N, K), dtype=torch.bfloat16, device='cuda') * 1
    # W = torch.eye(K, dtype=torch.bfloat16, device='cuda') * 1
    # reorder_index = torch.randperm(K, dtype=torch.int16, device='cuda')
    reorder_index = torch.arange(K, dtype=torch.int16, device='cuda') 

    # col_abs_max_values = torch.max(torch.abs(X), dim=0).values # .values 获取最大值本身
    # _, reorder_index_long = torch.sort(col_abs_max_values, descending=False)
    # reorder_index = reorder_index_long.to(torch.int16)
    WT = W.t().clone()
    AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(X, reorder_index, KN, KS, KO)
    BN, BS, BO, SFBN, SFBS, SFBO = mixedgemm.reorder_quantize_w(W, reorder_index, KN, KS, KO)

    # print("--- Outputs from reorder_quantize_x ---")
    # outputs_x = {"AN": AN, "AS": AS, "AO": AO, "SFAN": SFAN, "SFAS": SFAS, "SFAO": SFAO}
    # for name, tensor_val in outputs_x.items():
    #     print(f"{name}: shape={tensor_val.shape}, dtype={tensor_val.dtype}")
    #     if torch.is_floating_point(tensor_val) or tensor_val.dtype == torch.bfloat16: # Only check float/bf16 for nan/inf
    #         print(f"  {name} has nan: {torch.isnan(tensor_val.float()).any().item()}") # Convert to float32 for isnan if needed
    #         print(f"  {name} has inf: {torch.isinf(tensor_val.float()).any().item()}")
    #         # print(f"  {name} sample (float32 view): {tensor_val.float().flatten()[:10]}") # View as float32 to see values
    #     else: # For uint8 tensors, print raw values
    #         print(f"  {name} sample (uint8): {tensor_val.flatten()[:10]}")
    # print("--- Outputs from reorder_quantize_w ---")
    # outputs_x = {"BN": BN, "BS": BS, "BO": BO, "SFBN": SFBN, "SFBS": SFBS, "SFBO": SFBO}
    # for name, tensor_val in outputs_x.items():
    #     print(f"{name}: shape={tensor_val.shape}, dtype={tensor_val.dtype}")
    #     if torch.is_floating_point(tensor_val) or tensor_val.dtype == torch.bfloat16: # Only check float/bf16 for nan/inf
    #         print(f"  {name} has nan: {torch.isnan(tensor_val.float()).any().item()}") # Convert to float32 for isnan if needed
    #         print(f"  {name} has inf: {torch.isinf(tensor_val.float()).any().item()}")
    #         # print(f"  {name} sample (float32 view): {tensor_val.float().flatten()[:10]}") # View as float32 to see values
    #     else: # For uint8 tensors, print raw values
    #         print(f"  {name} sample (uint8): {tensor_val.flatten()[:10]}")

    C = mixedgemm.matmul(AN, BN, AS, BS, AO, BO, SFAN, SFBN, SFAS, SFBS, SFAO, SFBO)

    # D = torch.matmul(X[:, :2560], W[:2560,:])
    D = torch.matmul(X.to(torch.float32), WT.to(torch.float32))

    # print("输出张量 C 的形状:", C.shape)
    # print("输出张量 C 的数据类型:", C.dtype)

    # print("输出张量 D 的形状:", D.shape)
    # print("输出张量 D 的数据类型:", D.dtype)

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
    print(f"误差E: {mse_alt.item() / 1e6:.6f}")
    print(f"finish {i}")
    time.sleep(1)