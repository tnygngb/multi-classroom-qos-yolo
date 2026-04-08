# multi-classroom-qos-yolo

多教室多视频流并发课堂检测系统（阶段化实现）。

当前阶段：
- 阶段 0：仓库/配置/日志初始化完成
- 阶段 1：单流 baseline 训练与推理链路完成
- 阶段 2：baseline 与 enhanced（高分辨率小目标方案）双模型切换与对比完成

## 环境安装（Conda）

```bash
conda create -n multi-classroom-qos-yolo python=3.10 -y
conda activate multi-classroom-qos-yolo
pip install -r requirements.txt
```

## 阶段 1/2 入口

### 训练（baseline / enhanced）

```bash
# baseline
python -m scripts.train_detector --variant baseline

# enhanced
python -m scripts.train_detector --variant enhanced

# 指定配置文件（优先级高于 --variant）
python -m scripts.train_detector --config configs/detector/yolov12_p2_head.yaml
```

可先 dry-run 检查：

```bash
python -m scripts.train_detector --variant baseline --dry-run
```

### 单流推理（图片 / 视频 / 摄像头）

```bash
# baseline
python -m scripts.infer_single --variant baseline --source data/demos/room_101.mp4 --stream-id room_101

# enhanced
python -m scripts.infer_single --variant enhanced --source data/demos/room_101.mp4 --stream-id room_101

# 摄像头
python -m scripts.infer_single --variant baseline --source 0 --stream-id room_cam --max-frames 300
```

### 阶段 2 单流对比（mAP / Params / FLOPs / FPS）

```bash
# 如果暂时没有验证集，可跳过 mAP 验证
python -m scripts.benchmark_single --skip-val

# 使用视频样例测 FPS
python -m scripts.benchmark_single --source data/demos/room_101.mp4 --measure-frames 120 --skip-val
```

输出：
- 对比报告 JSON：`outputs/reports/single_stream_compare_*.json`
- 对比报告 CSV：`outputs/reports/single_stream_compare_*.csv`

## 配置说明

- 基础配置：`configs/base.yaml`
- baseline：`configs/detector/yolov8_head.yaml`
- enhanced：`configs/detector/yolov12_p2_head.yaml`
- 数据集描述：`data/annotations/head_dataset.yaml`

说明：
- 所有路径从配置读取，不需要写死本机绝对路径。
- 当配置请求 `cuda` 但环境不可用时，程序会自动回退到 `cpu` 并记录日志。

## 输出目录

- 日志：`outputs/logs/`
- 训练输出：`outputs/runs/`
- 推理视频/图片：`outputs/videos/`
- 指标与报告：`outputs/metrics/`, `outputs/reports/`
