# V1-R.2F CPU/CUDA Parity

状态：`blocked_missing_results`。请求冻结场景：10。

`torch.cuda.is_available()`：`True`；CUDA device count：`8`；`nvidia-smi` return code：`0`。

正式评测设备协议：`cuda`；冻结：`True`。

阻塞原因：
- both --cpu-results and --cuda-results are required
