# V1-R.2F CPU/CUDA Parity

状态：`blocked_unavailable_cuda`。请求冻结场景：10。

`torch.cuda.is_available()`：`False`；CUDA device count：`0`；`nvidia-smi` return code：`9`。

正式评测设备协议：`cpu`；冻结：`True`。

阻塞原因：
- torch.cuda.is_available() is false
- nvidia-smi cannot access an NVIDIA driver
