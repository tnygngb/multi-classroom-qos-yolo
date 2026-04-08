
# 多教室多视频流并发课堂检测项目实施文档（给 ChatGPT-5.3 Codex 的执行版）

## 文档用途
这份文档的目标不是解释概念，而是把整个项目拆成一套可以直接执行的实施流程。你可以把本文完整丢给 Codex，让它严格按阶段推进，每完成一阶段就提交代码、自测并汇报结果。

## 项目目标
构建一个“面向多教室、多视频流、多并发”的课堂检测系统。系统运行在一台边缘节点机器上，接入多路教室视频流，完成实时人头检测或课堂行为检测，并根据每路视频流的复杂度、人数密度、异常概率和系统负载，动态调整抽帧率、分辨率、模型版本和是否上传云端复核。

## 第一版范围（必须严格收缩）
第一版只做以下范围，不允许 Codex 擅自扩展：
1. 任务优先级：
   - 第一优先级：人头检测
   - 第二优先级：4 类课堂行为检测（可选）
2. 并发规模：
   - 先支持 4 路视频流
   - 再扩展到 8 路视频流
3. 运行环境：
   - 单机
   - 单 GPU（8GB 显存）
4. 模型路线：
   - baseline：YOLOv8n / YOLOv8s 或现有 YOLOv12 基线
   - improved：加入 P2 或高分辨率小目标分支
5. 调度策略：
   - 第一版只实现启发式 QoS 调度
   - 不做强化学习
6. 云端能力：
   - 只做低置信度帧复核与异常片段上传
   - 不做复杂训练平台
7. 界面：
   - 第一版不做复杂前端
   - 仅提供日志、JSON 输出、简单可视化视频和 FastAPI 接口

## 给 Codex 的总执行规则
Codex 必须遵守以下规则：
1. 不允许一次性生成整套系统后再一起调试，必须按阶段推进。
2. 每一阶段完成后，必须先自测，再进入下一阶段。
3. 每一阶段都要输出：
   - 修改了哪些文件
   - 当前可以运行的命令
   - 当前阶段的验收结果
   - 下一阶段计划
4. 不允许为了“先跑通”而写大量硬编码路径。
5. 所有参数必须配置化，不允许散落在脚本里。
6. 所有模块必须带最基础的日志输出和异常处理。
7. 如果遇到显存不足，优先降低：
   - 输入分辨率
   - 抽帧率
   - 并发流数
   - 模型规模
   而不是直接放弃多流框架。
8. 默认使用 Python 实现边缘节点、调度器、云端服务；能复用 Ultralytics 的地方尽量复用，但不能把业务逻辑全部堆进一个脚本。
9. 任何时候都不要同时开启“新模型设计 + 多流调度 + 云端复核 + 可视化前端”四条战线。

## 推荐项目名称
建议仓库名：
`multi-classroom-qos-yolo`

建议系统标题：
“多教室多视频流并发课堂检测系统”

## 建议技术栈
- Python 3.10+
- PyTorch
- Ultralytics
- OpenCV
- FastAPI
- Uvicorn
- Pydantic
- asyncio
- queue / threading / multiprocessing（按模块选择）
- Redis（可选；第一版可不用）
- SQLite（第一版可用）
- matplotlib（实验出图）
- pandas（结果整理）
- PyYAML（配置）
- loguru 或 logging

## 最终目录结构（目标状态）
```text
multi-classroom-qos-yolo/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  ├─ base.yaml
│  ├─ detector/
│  │  ├─ yolov8_head.yaml
│  │  ├─ yolov12_p2_head.yaml
│  ├─ streams/
│  │  ├─ demo_4streams.yaml
│  │  ├─ demo_8streams.yaml
│  ├─ scheduler/
│  │  ├─ qos_policy.yaml
│  ├─ cloud/
│  │  ├─ cloud_review.yaml
│  └─ experiments/
│     ├─ single_stream.yaml
│     ├─ multi_stream.yaml
├─ scripts/
│  ├─ train_detector.py
│  ├─ infer_single.py
│  ├─ run_edge_node.py
│  ├─ run_cloud_service.py
│  ├─ benchmark_single.py
│  ├─ benchmark_multi.py
│  ├─ export_results.py
│  └─ visualize_results.py
├─ src/
│  ├─ common/
│  │  ├─ config.py
│  │  ├─ logger.py
│  │  ├─ time_utils.py
│  │  ├─ io_utils.py
│  │  └─ types.py
│  ├─ detector/
│  │  ├─ model_loader.py
│  │  ├─ predictor.py
│  │  ├─ postprocess.py
│  │  ├─ tracker.py
│  │  └─ schemas.py
│  ├─ streams/
│  │  ├─ reader.py
│  │  ├─ frame_buffer.py
│  │  ├─ sampler.py
│  │  ├─ stream_state.py
│  │  └─ source_manager.py
│  ├─ scheduler/
│  │  ├─ qos_scheduler.py
│  │  ├─ scoring.py
│  │  ├─ policies.py
│  │  └─ dispatch.py
│  ├─ edge/
│  │  ├─ node.py
│  │  ├─ worker_pool.py
│  │  ├─ batching.py
│  │  ├─ result_sink.py
│  │  └─ event_engine.py
│  ├─ cloud/
│  │  ├─ app.py
│  │  ├─ review_service.py
│  │  ├─ analytics.py
│  │  └─ sample_bank.py
│  ├─ metrics/
│  │  ├─ latency.py
│  │  ├─ throughput.py
│  │  ├─ fairness.py
│  │  ├─ gpu_stats.py
│  │  └─ recorder.py
│  └─ api/
│     ├─ schemas.py
│     └─ routers.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ annotations/
│  └─ demos/
├─ outputs/
│  ├─ logs/
│  ├─ runs/
│  ├─ metrics/
│  ├─ videos/
│  └─ reports/
└─ tests/
   ├─ test_config.py
   ├─ test_sampler.py
   ├─ test_scheduler.py
   ├─ test_frame_buffer.py
   └─ test_api.py
```

## 实施阶段总览
项目必须按以下 10 个阶段推进：

1. 阶段 0：仓库与环境初始化  
2. 阶段 1：单流检测 baseline 跑通  
3. 阶段 2：单流检测增强版跑通  
4. 阶段 3：多流读流与缓冲模块实现  
5. 阶段 4：共享推理 worker 与固定策略多流并发  
6. 阶段 5：QoS 调度器实现  
7. 阶段 6：事件引擎与云端复核接口  
8. 阶段 7：日志、指标记录与实验脚本  
9. 阶段 8：对比实验与论文图表输出  
10. 阶段 9：代码整理、测试补全、交付文档整理  

---

# 阶段 0：仓库与环境初始化

## 目标
建立可维护的仓库结构，准备好最小可运行环境，统一配置加载和日志系统。

## 具体任务
1. 初始化 git 仓库。
2. 创建 `dev` 分支。
3. 生成基础目录结构。
4. 编写 `requirements.txt`。
5. 编写 `.gitignore`。
6. 实现最小配置系统 `src/common/config.py`。
7. 实现统一日志模块 `src/common/logger.py`。
8. 编写 `README.md`，说明项目目标与运行方式。
9. 准备 `configs/base.yaml`。

## `requirements.txt` 建议初稿
```text
torch
torchvision
ultralytics
opencv-python
fastapi
uvicorn
pydantic
pyyaml
pandas
matplotlib
numpy
loguru
psutil
pytest
```

## `configs/base.yaml` 最低要求
必须包含：
- 项目名称
- 默认输出目录
- 默认设备（cuda/cpu）
- 默认日志级别
- 随机种子
- 是否开启云端复核

示例结构：
```yaml
project_name: multi-classroom-qos-yolo
seed: 42
device: cuda
log_level: INFO
output_dir: outputs
enable_cloud_review: true
```

## 验收标准
- `python -m scripts.benchmark_single --help` 这类入口脚本可以成功解析参数（即使功能未实现）
- 配置文件可正常加载
- 日志模块可输出到控制台和文件
- 仓库目录结构完整

## 给 Codex 的执行提示词
请先完成阶段 0。要求：
1. 创建完整仓库骨架。
2. 实现配置加载工具和统一日志工具。
3. 写出 requirements.txt、.gitignore、README.md、base.yaml。
4. 不要开始写检测模型逻辑。
5. 完成后告诉我：
   - 新建了哪些文件
   - 关键模块如何运行
   - 如何验证阶段 0 已完成

---

# 阶段 1：单流检测 baseline 跑通

## 目标
先让系统具备稳定的单流推理能力。不要一开始就做并发。

## 推荐策略
1. 先做“人头检测”版本。
2. 使用已有数据集或你现有数据，整理成 Ultralytics 标准格式。
3. 优先跑通：
   - 训练脚本
   - 单图推理
   - 单视频推理
4. 基线模型优先：
   - YOLOv8n 或 YOLOv8s
   - 如果已有 YOLOv12 基线，也可以用，但必须确保训练脚本稳定

## 需要实现的文件
- `scripts/train_detector.py`
- `scripts/infer_single.py`
- `src/detector/model_loader.py`
- `src/detector/predictor.py`
- `src/detector/postprocess.py`
- `src/detector/schemas.py`

## 功能要求
1. `train_detector.py`
   - 读取 detector 配置
   - 调用 Ultralytics 完成训练
   - 将日志和权重保存到 `outputs/runs/`

2. `infer_single.py`
   - 支持输入图片、视频、本地摄像头
   - 输出带框结果视频或图片
   - 输出 JSON 检测结果

3. `model_loader.py`
   - 统一加载模型
   - 支持按配置选择模型路径和设备

4. `predictor.py`
   - 封装推理入口
   - 输入单帧，输出标准化检测结果

5. `postprocess.py`
   - 标准化框格式
   - 过滤低置信度结果
   - 转换为统一 schema

## 验收标准
- 单图推理可跑通
- 单视频推理可跑通
- 输出可视化视频
- 输出 JSON 检测结果
- 记录单流 FPS 与单流平均延迟

## Codex 注意事项
- 不要把 Ultralytics 调用散落在多个脚本中
- 必须封装 predictor 层
- 所有路径从配置读取
- 模型路径不存在时要给出清晰错误信息

## 给 Codex 的执行提示词
请实现阶段 1：单流检测 baseline 跑通。要求：
1. 完成训练脚本、推理脚本、模型加载器、预测器和后处理模块。
2. 支持图片和视频推理。
3. 推理结果同时保存可视化文件和 JSON。
4. 输出单流平均延迟和 FPS。
5. 完成后列出运行命令和示例配置。

---

# 阶段 2：单流检测增强版跑通

## 目标
在 baseline 基础上实现“更适合课堂小目标”的增强版检测器，但增强范围必须克制。

## 建议增强项
第一版最多做 1 到 2 个改动：
1. P2 分支或高分辨率小目标分支
2. 一个轻量注意力/特征融合模块

不允许一下子堆四五个模块。

## 需要做的事情
1. 为增强模型准备独立配置：
   - `configs/detector/yolov12_p2_head.yaml`
2. 确保增强模型仍然能被统一的 `model_loader.py` 加载
3. 在训练与推理脚本中增加“选择 baseline / enhanced”的能力
4. 输出对比结果：
   - mAP@0.5
   - mAP@0.5:0.95
   - Params
   - FLOPs（如果方便）
   - FPS

## 这一阶段的原则
目标不是把模型做到最复杂，而是获得一个“单流有效、后续可部署到多流系统中”的 detector。

## 验收标准
- baseline 与 enhanced 两个模型都能训练和推理
- 能导出一张单流对比表
- enhanced 模型没有严重牺牲速度
- 能在你的视频样例上看到对小目标更稳定的效果

## 给 Codex 的执行提示词
请实现阶段 2：单流检测增强版。要求：
1. 在 baseline 基础上增加 P2 或高分辨率分支。
2. 保持与已有训练、推理接口兼容。
3. 新模型必须通过统一 loader 加载。
4. 输出 baseline 与 enhanced 的性能对比结果。
5. 不要在这一阶段实现多流或调度。

---

# 阶段 3：多流读流与缓冲模块实现

## 目标
先实现“多路视频进来”，暂时不考虑动态调度。

## 这一阶段的核心思想
把视频系统拆成：
- 读流
- 缓冲
- 抽帧
- 推理
- 输出

不要在一个循环里把这些事情全部写死。

## 需要实现的文件
- `src/streams/reader.py`
- `src/streams/frame_buffer.py`
- `src/streams/sampler.py`
- `src/streams/stream_state.py`
- `src/streams/source_manager.py`
- `configs/streams/demo_4streams.yaml`

## 设计要求

### 1. reader.py
职责：
- 打开单路视频源
- 持续读取帧
- 为每一帧打时间戳
- 将帧写入对应 buffer

支持输入：
- 本地 mp4
- RTSP
- 摄像头编号

### 2. frame_buffer.py
职责：
- 每路流维护一个固定容量缓冲区
- 新帧来了就写入
- 队列满时丢弃旧帧或按策略丢帧
- 提供读取最新帧、读取采样帧的接口

建议实现：
- `push(frame, ts)`
- `get_latest()`
- `get_batch(n)`
- `size()`
- `drop_count`

### 3. sampler.py
职责：
- 决定每路流当前取帧频率
- 第一版先做固定采样率
- 后续由调度器动态调整

### 4. stream_state.py
职责：
为每一路流维护状态字段：
- stream_id
- source
- online/offline
- buffer_size
- drop_count
- last_frame_ts
- current_sampling_fps
- last_infer_latency
- last_detect_count

### 5. source_manager.py
职责：
- 根据 `demo_4streams.yaml` 创建多个 reader
- 管理多路流生命周期

## `demo_4streams.yaml` 示例
```yaml
streams:
  - stream_id: room_101
    source: data/demos/room_101.mp4
  - stream_id: room_102
    source: data/demos/room_102.mp4
  - stream_id: room_103
    source: data/demos/room_103.mp4
  - stream_id: room_104
    source: data/demos/room_104.mp4
sampling_fps: 5
buffer_size: 32
```

## 验收标准
- 4 路视频能同时打开
- 每路流都有独立 buffer
- 可以实时输出各流 buffer 状态
- 当某一路掉线时，不影响其他路继续工作
- 能打印每一路当前读取帧率和丢帧统计

## 给 Codex 的执行提示词
请实现阶段 3：多流读流与缓冲模块。要求：
1. 支持 4 路视频流同时接入。
2. 每路流使用独立 frame buffer。
3. 支持固定采样率。
4. 要有 stream state 结构保存实时状态。
5. 输出每路流的在线状态、buffer 大小和丢帧数。
6. 暂时不要接入推理模型。

---

# 阶段 4：共享推理 worker 与固定策略多流并发

## 目标
让多路流真正进入共享推理流程，但先不做动态调度。

## 这一阶段的关键
先实现：
- 多流读帧
- 固定采样
- 统一推理入口
- 共享 detector worker
- 结果输出

不要先做复杂优化。

## 需要实现的文件
- `src/edge/node.py`
- `src/edge/worker_pool.py`
- `src/edge/batching.py`
- `src/edge/result_sink.py`
- `scripts/run_edge_node.py`

## 设计说明

### 1. node.py
职责：
- 启动所有 streams
- 定期从 buffer 中取采样帧
- 投递给推理 worker
- 收集结果并输出

### 2. worker_pool.py
职责：
- 管理一个或多个共享 detector worker
- 避免“一路流开一个模型实例”
- 控制最大并发推理数量

### 3. batching.py
职责：
- 可选地把多个流的待推理帧组合成 batch
- 如果 batch 难以稳定，第一版可以先单帧推理，但代码结构要预留

### 4. result_sink.py
职责：
- 将检测结果写到 JSONL
- 可选写入 CSV
- 可选保存带框视频

## 运行逻辑
每个调度周期：
1. 从每路流的 buffer 中获取一帧
2. 送入统一待推理队列
3. 由 worker_pool 调用 predictor 执行推理
4. 将结果写回 stream_state
5. 记录延迟、目标数和队列长度

## 验收标准
- 4 路流可在固定配置下同时推理
- 系统能输出总吞吐与各流吞吐
- 系统能输出各流平均延迟
- 不因为单路异常而整体崩溃
- 显存占用可控

## 给 Codex 的执行提示词
请实现阶段 4：固定策略下的多流并发推理。要求：
1. 用共享 worker_pool 处理多路流推理。
2. 不允许每路流单独加载一个模型实例。
3. 结果写入 JSONL 或 CSV。
4. 输出总吞吐、各流吞吐、平均延迟。
5. 先不实现动态调度。

---

# 阶段 5：QoS 调度器实现

## 目标
在多流固定推理基础上，引入动态调度能力，让系统根据每路视频的重要性和负载状态自动调整策略。

## 第一版调度器只做启发式策略
不要实现强化学习。先实现三档模式：

- `ECO`
- `NORMAL`
- `ALERT`

## 需要实现的文件
- `src/scheduler/qos_scheduler.py`
- `src/scheduler/scoring.py`
- `src/scheduler/policies.py`
- `src/scheduler/dispatch.py`
- `configs/scheduler/qos_policy.yaml`

## 调度输入字段
每一路流至少读取以下指标：
- 当前人数估计
- 当前检测目标数量
- 最近窗口平均延迟
- 最近窗口丢帧率
- buffer 积压程度
- 异常行为得分（第一版可简单定义）

## 评分公式（第一版）
```text
score = a * density_score
      + b * motion_or_activity_score
      + c * anomaly_score
      + d * queue_pressure_score
```

其中参数 `a b c d` 来自配置文件。

## 模式定义建议

### ECO
- resolution = 640
- sampling_fps = 2~4
- model_variant = light

### NORMAL
- resolution = 960 或 1280
- sampling_fps = 5~8
- model_variant = main

### ALERT
- resolution = 更高
- sampling_fps = 8~12
- model_variant = enhanced
- 可触发云端复核

## `qos_policy.yaml` 示例
```yaml
scheduler_tick_sec: 2
weights:
  density: 0.35
  activity: 0.20
  anomaly: 0.25
  queue_pressure: 0.20

modes:
  ECO:
    input_size: 640
    sampling_fps: 3
    model_variant: light
    cloud_review: false
  NORMAL:
    input_size: 960
    sampling_fps: 6
    model_variant: main
    cloud_review: false
  ALERT:
    input_size: 1280
    sampling_fps: 10
    model_variant: enhanced
    cloud_review: true

thresholds:
  eco_max: 0.35
  normal_max: 0.70
```

## 调度器行为
每隔 `scheduler_tick_sec` 秒：
1. 读取所有 stream_state
2. 为每路流打分
3. 根据分数决定模式
4. 更新 sampler、模型选择和是否触发云端复核
5. 记录本轮调度决策日志

## 验收标准
- 系统能定期输出每路流当前模式
- 模式切换可实际影响采样率、分辨率、模型选择
- 系统在负载变高时会自动降低低优先级流配置
- 有完整决策日志可追踪

## 给 Codex 的执行提示词
请实现阶段 5：QoS 调度器。要求：
1. 用启发式评分实现 ECO/NORMAL/ALERT 三档模式。
2. 调度器能周期性读取各路 stream_state。
3. 调度结果必须真正影响采样率、输入尺寸、模型版本和云端复核开关。
4. 所有调度决策都要写入日志。
5. 不要实现强化学习版本。

---

# 阶段 6：事件引擎与云端复核接口

## 目标
把“低置信度帧、异常片段上传云端复核”的链路打通。

## 需要实现的文件
- `src/edge/event_engine.py`
- `src/cloud/app.py`
- `src/cloud/review_service.py`
- `src/cloud/sample_bank.py`
- `src/cloud/analytics.py`
- `scripts/run_cloud_service.py`

## 边缘端 event_engine 负责
1. 根据检测结果判断是否触发事件
2. 事件类型建议包括：
   - low_confidence_detection
   - abnormal_activity_suspected
   - dense_crowd_frame
3. 选取关键帧或短片段上传
4. 接收云端复核结果并回写

## 云端服务 API 建议
### POST `/review/frame`
输入：
- stream_id
- timestamp
- image
- edge_prediction

输出：
- refined_prediction
- confidence
- action

### POST `/review/clip`
输入：
- stream_id
- start_ts
- end_ts
- clip file

输出：
- event_type
- confidence
- notes

### POST `/metrics/summary`
输入：
- 边缘节点统计摘要

输出：
- ack

## 第一版 sample_bank.py 负责
- 保存被上传的关键帧路径
- 保存边缘预测和云端预测
- 保存复核时间

## analytics.py 第一版只需完成
- 汇总各教室事件次数
- 汇总各教室活跃度统计
- 生成简单 JSON 报告

## 验收标准
- 边缘端能够向云端发送低置信度帧
- 云端能够返回复核结果
- 复核结果可记录到 sample bank
- 整条链路失败时不影响边缘端主流程

## 给 Codex 的执行提示词
请实现阶段 6：事件引擎与云端复核接口。要求：
1. 在边缘端新增 event_engine。
2. 用 FastAPI 实现云端复核服务。
3. 支持单帧复核和片段复核接口。
4. 复核结果保存到 sample_bank。
5. 网络失败不能阻塞边缘主流程。

---

# 阶段 7：日志、指标记录与实验脚本

## 目标
让系统变成“可测量、可比较、可复现实验”的状态。

## 需要实现的文件
- `src/metrics/latency.py`
- `src/metrics/throughput.py`
- `src/metrics/fairness.py`
- `src/metrics/gpu_stats.py`
- `src/metrics/recorder.py`
- `scripts/benchmark_single.py`
- `scripts/benchmark_multi.py`
- `scripts/export_results.py`
- `scripts/visualize_results.py`

## 必须记录的指标
### 单流指标
- FPS
- 平均延迟
- p95 延迟
- 单流 mAP（离线评估）

### 多流指标
- 总吞吐（总 FPS）
- 各流吞吐
- 各流平均延迟
- p95 延迟
- buffer 平均积压
- 丢帧率
- 显存占用
- GPU 利用率
- 流间公平性

## 公平性建议
第一版可以实现一个简单指标，例如：
- 各流吞吐标准差
- Jain’s fairness index（如果想做得更规范）

## `benchmark_multi.py` 要支持的场景
- 1 路流
- 2 路流
- 4 路流
- 8 路流

并且支持三种策略对比：
1. 静态高配置
2. 静态低配置
3. QoS 调度

## 验收标准
- 可以一条命令跑多流 benchmark
- 能自动输出 CSV/JSON 报告
- 能自动画出：
   - 流数 vs 吞吐
   - 流数 vs 延迟
   - 流数 vs 有效检测性能

## 给 Codex 的执行提示词
请实现阶段 7：实验脚本与指标系统。要求：
1. 补齐 latency、throughput、fairness、gpu_stats 和 recorder 模块。
2. 实现 benchmark_single.py 与 benchmark_multi.py。
3. 输出 CSV/JSON 报告。
4. 实现 visualize_results.py，自动绘图。
5. 图表要能直接用于论文初稿。

---

# 阶段 8：对比实验与论文图表输出

## 目标
把系统跑成一篇论文需要的实验结果。

## 必做实验 A：单流模型对比
比较：
- baseline
- enhanced

指标：
- mAP@0.5
- mAP@0.5:0.95
- FPS
- Params
- 显存占用

## 必做实验 B：多流并发能力对比
比较：
- 静态低配置
- 静态高配置
- QoS 调度

并发流数：
- 1
- 2
- 4
- 8

指标：
- 总吞吐
- 各流平均延迟
- p95 延迟
- 丢帧率
- GPU 利用率

## 必做实验 C：跨教室泛化
建议：
- 用不同教室视频分 train/test
- 或者至少做“未见过教室”的测试

指标：
- 跨教室 mAP
- 性能下降率
- 误检与漏检分析

## 必做实验 D：消融实验
至少做：
1. 不加 P2 / 加 P2
2. 不加调度 / 加调度
3. 不启用云端复核 / 启用云端复核

## 论文图表建议输出
1. 系统总体架构图
2. detector 结构图
3. 多流调度流程图
4. 并发流数 vs 吞吐曲线
5. 并发流数 vs 延迟曲线
6. 调度模式分配示意图
7. 可视化检测样例图

## 给 Codex 的执行提示词
请实现阶段 8：实验组织与论文图表输出。要求：
1. 提供统一的实验脚本入口。
2. 自动生成表格和图表文件。
3. 支持单流、多流、消融、跨教室四类实验。
4. 图表输出到 outputs/reports/。
5. 每个实验结果都要有可复现实验配置记录。

---

# 阶段 9：代码整理、测试补全、交付文档整理

## 目标
把项目从“能跑”整理到“可交付”。

## 需要完成的事情
1. 补齐单元测试：
   - config
   - frame_buffer
   - sampler
   - scheduler
   - api
2. 统一命名与注释风格
3. 删除无用脚本
4. 补齐 README：
   - 项目简介
   - 环境安装
   - 训练方法
   - 单流运行方法
   - 多流运行方法
   - 云端服务运行方法
   - benchmark 方法
5. 输出一个 `docs/` 目录（如果愿意）
6. 准备演示命令

## README 至少包含的命令
```bash
python scripts/train_detector.py --config configs/detector/yolov8_head.yaml
python scripts/infer_single.py --config configs/detector/yolov8_head.yaml --source data/demos/room_101.mp4
python scripts/run_edge_node.py --config configs/streams/demo_4streams.yaml
python scripts/run_cloud_service.py --config configs/cloud/cloud_review.yaml
python scripts/benchmark_multi.py --config configs/experiments/multi_stream.yaml
```

## 验收标准
- 从新环境安装到运行 demo 有清晰步骤
- 关键模块都有最小测试
- 主流程不依赖手工改源码路径
- 输出目录结构清晰

## 给 Codex 的执行提示词
请实现阶段 9：整理与交付。要求：
1. 补齐最关键的单元测试。
2. 整理 README 与运行命令。
3. 清理无用代码和硬编码路径。
4. 输出一个可直接演示的版本。
5. 告诉我从零开始运行 demo 的完整步骤。

---

# 每个核心模块的详细设计要求

## A. detector 模块
### 输入
- 单帧图像
- 模型配置
- 推理参数

### 输出
统一格式的检测结果：
```json
{
  "stream_id": "room_101",
  "timestamp": 1711111111.123,
  "detections": [
    {
      "label": "head",
      "confidence": 0.91,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "latency_ms": 18.5
}
```

### 设计要求
- 模型加载只能在初始化时发生，不要每帧重复加载
- 预测接口要尽量纯净
- 后处理独立封装
- label 映射从配置读取

## B. frame_buffer 模块
### 关键要求
- 必须有容量上限
- 满了之后必须能定义丢帧策略
- 获取最新帧必须是 O(1) 或近似高效
- 线程安全

## C. scheduler 模块
### 最重要的要求
- 调度决策必须可解释
- 每次决策都要记录：
  - 输入状态
  - 分数
  - 模式
  - 采取的动作

## D. cloud 模块
### 要求
- 失败不能拖垮边缘端
- API 输入输出清晰
- 尽量异步化
- 样本记录可追踪

## E. metrics 模块
### 要求
- 所有时间单位统一
- 记录窗口长度可配置
- 结果可导出为 CSV/JSON
- 支持后续论文画图

---

# 建议的开发顺序（非常重要）

Codex 必须严格按这个顺序执行，不得跳步：

1. 先完成阶段 0  
2. 再完成阶段 1  
3. 再完成阶段 2  
4. 再完成阶段 3  
5. 再完成阶段 4  
6. 再完成阶段 5  
7. 再完成阶段 6  
8. 再完成阶段 7  
9. 再完成阶段 8  
10. 最后完成阶段 9  

任何时候都不允许跳过：
- 单流稳定验证
- 多流固定策略验证
- 调度器验证

---

# 推荐的提交节奏

建议每个阶段至少一个 commit，命名示例：
- `init: bootstrap repository structure and config system`
- `feat: add single-stream detector baseline`
- `feat: add enhanced detector with p2 branch`
- `feat: add multi-stream readers and frame buffers`
- `feat: add shared inference worker pool`
- `feat: add qos scheduler`
- `feat: add cloud review service`
- `feat: add benchmark and metrics pipeline`
- `chore: clean up project for delivery`

---

# 常见坑与处理原则

## 1. 显存炸掉
优先处理顺序：
1. 降低输入尺寸
2. 降低采样率
3. 减少并发流数
4. 换更小模型
5. 关闭 enhanced 模型

## 2. 多流延迟越来越高
优先检查：
- buffer 是否无限增长
- 是否每路流都试图全帧处理
- 是否每路流都独占模型实例
- 是否 worker 数量过多导致反而争抢

## 3. 路径混乱
解决原则：
- 所有路径只从配置读取
- 禁止在脚本里写本机绝对路径

## 4. 调度器没有效果
优先检查：
- 模式切换后是否真的改变了采样率和输入尺寸
- scoring 是否所有流都打成差不多的分
- thresholds 是否设置过死

## 5. 云端复核拖慢边缘端
解决原则：
- 上传必须异步
- 云端失败要快速返回
- 边缘端只保留关键摘要，不等待完整复核

---

# 交给 Codex 的总提示词（最终版）

下面这段可以直接复制给 Codex 作为总任务说明：

你将帮助我逐阶段实现一个“多教室多视频流并发课堂检测系统”。请严格遵守以下规则：

1. 必须严格按阶段推进，不允许一次性生成整套系统。
2. 每完成一个阶段后，必须：
   - 列出新增或修改的文件
   - 给出运行命令
   - 给出验收方式
   - 说明下一阶段计划
3. 第一版项目范围严格限制为：
   - 单机、单 GPU（8GB）
   - 先支持 4 路，再扩展到 8 路
   - 先做人头检测，再扩展到 4 类课堂行为
   - 先做启发式 QoS 调度，不做强化学习
   - 只做简单云端复核，不做复杂前端
4. 技术栈使用 Python、PyTorch、Ultralytics、OpenCV、FastAPI。
5. 所有配置必须写入 configs/，不得硬编码路径。
6. 所有模块要有基础日志和异常处理。
7. 项目必须形成以下能力：
   - 单流 detector baseline
   - enhanced detector
   - 多流读流和缓冲
   - 共享推理 worker_pool
   - QoS 调度器
   - 云端复核接口
   - benchmark 与可视化图表输出
8. 你必须先完成阶段 0，然后等待我确认或继续下一阶段。

---

# 附录：我建议你先让 Codex 做到的“最小闭环”

如果你希望更稳，可以先只让 Codex 做到以下闭环：
1. 单流人头检测 baseline
2. 4 路视频读流与 buffer
3. 共享 worker 固定策略推理
4. 输出每路流延迟、吞吐与 JSON 结果

等这个闭环稳定后，再继续：
5. QoS 调度
6. 云端复核
7. benchmark

这个顺序最不容易失控。

# 结束语
这份文档的使用方式很简单：
- 你可以把全文直接发给 Codex
- 也可以一次只发一个“阶段”给 Codex
- 最稳的方式是：按阶段执行，每阶段完成后让它先自检、再继续
