# multi-classroom-qos-yolo

多教室多视频流并发课堂检测系统（阶段化实现版本）。

当前仓库已完成**阶段 0：仓库与环境初始化**，包含完整项目骨架、统一配置加载工具与统一日志工具。检测模型与推理逻辑尚未开始实现。

## 1. 目标

- 单机单 GPU（8GB）运行
- 先支持 4 路视频流，再扩展到 8 路
- 先做人头检测，再扩展课堂行为检测
- 引入启发式 QoS 调度与云端复核链路

## 2. 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. 阶段 0 可运行命令

```bash
python -m scripts.benchmark_single --help
python -m scripts.benchmark_single --config configs/base.yaml --dry-run
python -m scripts.benchmark_multi --config configs/experiments/multi_stream.yaml --dry-run
python -m scripts.run_edge_node --config configs/streams/demo_4streams.yaml --dry-run
python -m scripts.run_cloud_service --config configs/cloud/cloud_review.yaml --dry-run
```

说明：
- 这些命令当前只验证参数解析、配置加载、日志初始化与输出目录创建。
- 真实业务逻辑会在后续阶段按文档逐步补齐。

## 4. 统一配置与日志

- 配置加载模块：`src/common/config.py`
  - 支持加载 `configs/base.yaml`
  - 支持 base + override 深度合并
  - 支持自动创建输出目录并回填路径
- 日志模块：`src/common/logger.py`
  - 控制台 + 文件双通道输出
  - 支持按配置设置日志等级与日志目录
  - 日志默认写入 `outputs/logs/`

## 5. 目录说明

仓库已经创建了最终目标结构的骨架，包括：
- `configs/`：基础配置、流配置、调度配置、云端配置、实验配置
- `scripts/`：训练、推理、边缘节点、云端服务、benchmark、导出、可视化入口脚本
- `src/`：common / detector / streams / scheduler / edge / cloud / metrics / api 模块目录
- `data/`：raw / processed / annotations / demos
- `outputs/`：logs / runs / metrics / videos / reports
- `tests/`：阶段化测试入口
