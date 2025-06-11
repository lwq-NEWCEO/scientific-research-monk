import torch
print(torch.__version__)      # 打印 PyTorch 版本
print(torch.cuda.is_available()) # 检查 CUDA 是否可用
print(torch.version.cuda)     # 打印 PyTorch 使用的 CUDA 版本
print(torch.cuda.device_count()) # 打印可用的 GPU 数量
print(torch.cuda.get_device_name(0)) # 打印第一个 GPU 的名称 (通常是您的 RTX 3080)
exit() # 退出 Python 交互环境
