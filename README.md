# multi-classroom-qos-yolo

多教室多视频流并发课堂检测系统（阶段化实现）。

当前仓库已完成阶段 0-9 的可交付版本，重点能力包括：
- 单流训练/推理（baseline/enhanced）
- 多流采集、缓冲、固定频率采样
- 共享推理池与 QoS 调度（ECO/NORMAL/ALERT）
- 事件触发与云端复核接口
- 指标记录、报告导出、图表可视化
- 统一实验编排与阶段 9 交付整理

## 1. 项目简介

系统在边缘节点接入多路教室视频流，执行课堂检测并根据流状态动态分配算力。支持把低置信或异常片段上传到云端复核服务，并产出可复现实验报告与图表。

## 2. 环境安装（Anaconda）

```bash
conda create -n multi-classroom-qos-yolo python=3.10 -y
conda activate multi-classroom-qos-yolo
pip install -r requirements.txt
```

## 3. 训练方法

```bash
# baseline
python scripts/train_detector.py --variant baseline

# enhanced
python scripts/train_detector.py --variant enhanced
```

默认配置来源：
- `configs/base.yaml`
- `configs/detector/yolov8_head.yaml`
- `configs/detector/yolov12_p2_head.yaml`

## 4. 单流运行方法

```bash
# 视频文件推理
python scripts/infer_single.py --variant baseline --source data/demos/room_101.mp4 --stream-id room_101
python scripts/infer_single.py --variant enhanced --source data/demos/room_101.mp4 --stream-id room_101

# 摄像头推理
python scripts/infer_single.py --variant baseline --source 0 --stream-id room_cam --max-frames 300
```

## 5. 多流运行方法

```bash
# 构建/配置检查（不执行长时推理）
python scripts/run_edge_node.py --config configs/streams/demo_4streams.yaml --detector-config configs/detector/yolov8_head.yaml --dry-run

# 启用 QoS 调度
python scripts/run_edge_node.py --config configs/streams/demo_4streams.yaml --detector-config configs/detector/yolov8_head.yaml --workers 1 --duration-sec 15 --status-interval-sec 2

# 禁用 QoS（静态策略）
python scripts/run_edge_node.py --config configs/streams/demo_4streams.yaml --detector-config configs/detector/yolov8_head.yaml --disable-qos --workers 1 --duration-sec 15 --status-interval-sec 2
```

## 6. 云端服务运行方法

```bash
# 运行云端复核服务
python scripts/run_cloud_service.py --config configs/cloud/cloud_review.yaml

# 仅检查配置与启动参数
python scripts/run_cloud_service.py --config configs/cloud/cloud_review.yaml --dry-run
```

配合边缘端运行（默认开启事件引擎）：

```bash
python scripts/run_edge_node.py --config configs/streams/demo_4streams.yaml --detector-config configs/detector/yolov8_head.yaml --cloud-config configs/cloud/cloud_review.yaml --workers 1 --duration-sec 15 --status-interval-sec 2
```

## 7. benchmark 方法

```bash
# 单流对比
python scripts/benchmark_single.py --config configs/experiments/single_stream.yaml --skip-val

# 多流矩阵 benchmark
python scripts/benchmark_multi.py --config configs/experiments/multi_stream.yaml --workers 1

# 汇总导出
python scripts/export_results.py --input-dir outputs/reports

# 图表生成
python scripts/visualize_results.py --output-dir outputs/reports
```

## 8. 阶段 8 一键实验编排

```bash
# 一键执行完整实验
python scripts/run_experiments.py --config configs/experiments/stage8_paper.yaml --mode all

# 快速 smoke 版
python scripts/run_experiments.py --config configs/experiments/stage8_quick.yaml --mode all
```

## 9. 测试与验收

```bash
# 阶段 9 核心模块单测
pytest -q tests/test_config.py tests/test_frame_buffer.py tests/test_sampler.py tests/test_scheduler.py tests/test_api.py

# 全量测试
pytest -q
```

`pytest.ini` 已限制测试发现路径为 `tests/`，避免扫描输出目录干扰。

## 10. 输出目录

- `outputs/logs/`：运行日志
- `outputs/runs/`：训练与推理运行产物
- `outputs/metrics/`：指标文件
- `outputs/videos/`：可视化视频
- `outputs/reports/`：实验报告、JSONL、图表

## 11. 从零演示文档

完整命令链请看：`docs/DEMO.md`
