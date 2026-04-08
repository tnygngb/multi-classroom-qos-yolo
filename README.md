# multi-classroom-qos-yolo

多教室多视频流并发课堂检测系统（阶段化开发）。

当前状态：
- 阶段 0：已完成（仓库骨架、配置系统、日志系统）
- 阶段 1：已完成基础单流 detector 训练/推理链路（baseline）

## 环境安装

### Conda（推荐）

```bash
conda create -n multi-classroom-qos-yolo python=3.10 -y
conda activate multi-classroom-qos-yolo
pip install -r requirements.txt
```

## 关键目录

- `configs/`：配置文件
- `scripts/`：训练、推理与运行入口
- `src/common/`：配置与日志工具
- `src/detector/`：模型加载、预测、后处理、输出 schema
- `outputs/`：日志、结果、可视化输出

## 阶段 1 运行命令

### 1) 训练（baseline）

```bash
python -m scripts.train_detector --config configs/detector/yolov8_head.yaml
```

可先验证参数与配置：

```bash
python -m scripts.train_detector --config configs/detector/yolov8_head.yaml --dry-run
```

### 2) 单图推理

```bash
python -m scripts.infer_single --config configs/detector/yolov8_head.yaml --source data/demos/sample.jpg --stream-id room_101
```

### 3) 单视频推理

```bash
python -m scripts.infer_single --config configs/detector/yolov8_head.yaml --source data/demos/room_101.mp4 --stream-id room_101
```

### 4) 摄像头推理

```bash
python -m scripts.infer_single --config configs/detector/yolov8_head.yaml --source 0 --stream-id room_cam --max-frames 300
```

## 输出说明

- 推理可视化：默认写入 `outputs/videos/`
- 推理 JSON：默认写入 `outputs/reports/`
- 运行日志：写入 `outputs/logs/`

## 配置说明

- 基础配置：`configs/base.yaml`
- baseline detector 配置：`configs/detector/yolov8_head.yaml`
- enhanced 占位配置：`configs/detector/yolov12_p2_head.yaml`

默认训练数据描述文件为：
- `data/annotations/head_dataset.yaml`

你可以在 detector 配置中通过 `detector.train.data` 改成自己的数据集 YAML。
