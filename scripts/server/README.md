# 服务器执行脚本(租卡日专用)

所有在租用 GPU 服务器上运行的脚本统一放这里,规则:

1. 每个脚本可独立运行、幂等(重跑不炸、已完成的步骤自动跳过)
2. 大文件(模型/标签/sidecar)不进 git —— 从 OSS 恢复,校验和清单见 manifest/
3. 一切以固定版本运行:repo commit、VeloxMesh commit、vLLM 版本、模型 revision

## 执行顺序(租卡日)

| # | 脚本 | 作用 |
|---|---|---|
| 1 | restore_from_oss.sh | 下载并校验 checkpoint/标签/sidecar/workload |
| 2 | setup_env.sh | venv + vllm==0.24.x + 依赖 + PYTHONPATH |
| 3 | launch_predictor.sh | 起预测服务(gRPC :50052 / HTTP) |
| 4 | launch_gateway.sh | 克隆/编译/启动 VeloxMesh(固定 commit) |
| 5 | launch_vllm.sh | 起推理引擎(--scheduler-cls 按策略传入) |
| 6 | calibrate_saturation.sh | FCFS 饱和标定 → 得出各档请求速率 |
| 7 | run_benchmark_matrix.sh | 六策略外层循环(起→健康→runner→关→下一个) |

产出统一落 /hy-tmp/results/,跑完回传 OSS 再关机。
