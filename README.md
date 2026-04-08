# multi-classroom-qos-yolo

Staged implementation of a multi-classroom, multi-stream classroom detection system.

Current progress:
- Stage 0: repository/config/logger bootstrap
- Stage 1: single-stream baseline training/inference pipeline
- Stage 2: baseline vs enhanced variant switching and comparison benchmark
- Stage 3: multi-stream reading, buffering, fixed-rate sampling, stream state tracking
- Stage 4: shared inference worker pool with fixed-policy multi-stream dispatch
- Stage 5: heuristic QoS scheduler with ECO/NORMAL/ALERT dynamic mode control

## Setup (Conda)

```bash
conda create -n multi-classroom-qos-yolo python=3.10 -y
conda activate multi-classroom-qos-yolo
pip install -r requirements.txt
```

## Stage 1/2 commands

```bash
# Train (baseline / enhanced)
python -m scripts.train_detector --variant baseline
python -m scripts.train_detector --variant enhanced

# Inference (image/video/camera)
python -m scripts.infer_single --variant baseline --source data/demos/room_101.mp4 --stream-id room_101
python -m scripts.infer_single --variant enhanced --source data/demos/room_101.mp4 --stream-id room_101
python -m scripts.infer_single --variant baseline --source 0 --stream-id room_cam --max-frames 300

# Single-stream comparison table
python -m scripts.benchmark_single --skip-val
python -m scripts.benchmark_single --source data/demos/room_101.mp4 --measure-frames 120 --skip-val
```

Outputs:
- JSON: `outputs/reports/single_stream_compare_*.json`
- CSV: `outputs/reports/single_stream_compare_*.csv`

## Stage 3/4/5 commands

```bash
# Build-only check
python -m scripts.run_edge_node --config configs/streams/demo_4streams.yaml --detector-config configs/detector/yolov8_head.yaml --dry-run

# Run with stage-5 QoS scheduler
python -m scripts.run_edge_node --config configs/streams/demo_4streams.yaml --detector-config configs/detector/yolov8_head.yaml --workers 1 --duration-sec 15 --status-interval-sec 2

# Run without QoS scheduler (static profile)
python -m scripts.run_edge_node --config configs/streams/demo_4streams.yaml --detector-config configs/detector/yolov8_head.yaml --disable-qos --workers 1 --duration-sec 15 --status-interval-sec 2
```

Stage 3 behavior:
- Supports local files, RTSP, and camera index sources
- One independent fixed-capacity frame buffer per stream
- Fixed-rate sampler per stream (current version)
- Stream state fields include online/offline, buffer size, drop count, read FPS, and last frame timestamp
- A failing stream is isolated and does not crash other streams

Stage 4 behavior:
- Uses shared worker pool (`src/edge/worker_pool.py`) instead of one model per stream
- Dispatches sampled frames from all streams into a unified inference queue
- Writes inference results to JSONL and CSV via `src/edge/result_sink.py`
- Periodically logs total throughput, per-stream throughput, and average latency

Stage 5 behavior:
- Adds periodic heuristic scheduler (`src/scheduler/qos_scheduler.py`)
- Scores each stream from density/activity/anomaly/queue pressure signals
- Dynamically applies `ECO/NORMAL/ALERT` actions to sampling FPS, input size, model variant, and cloud-review switch
- Writes explainable scheduling logs with input state, score breakdown, mode, and action

## Main configs

- Base: `configs/base.yaml`
- Detector baseline: `configs/detector/yolov8_head.yaml`
- Detector enhanced: `configs/detector/yolov12_p2_head.yaml`
- Streams demo (4): `configs/streams/demo_4streams.yaml`
- Streams demo (8): `configs/streams/demo_8streams.yaml`
- Scheduler policy: `configs/scheduler/qos_policy.yaml`

## Output directories

- Logs: `outputs/logs/`
- Train runs: `outputs/runs/`
- Metrics: `outputs/metrics/`
- Videos: `outputs/videos/`
- Reports: `outputs/reports/`
