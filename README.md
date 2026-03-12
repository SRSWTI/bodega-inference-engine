# Bodega Inference Engine — Testing Suite

This repository contains robust tools, benchmarks, and interactive shells to test the full power of the Bodega Inference Engine running locally via `BodegaOS Sensors`.

## Getting Started

1. **Make the setup executable**:
   ```bash
   chmod +x setup.sh
   chmod +x install_sensors.sh
   ```

2. **Run the Interactive Setup Flow**:
   ```bash
   ./setup.sh
   ```
   > **Note:** The `setup.sh` script handles everything! It uses `install_sensors.sh` to download BodegaOS Sensors locally, guides you through turning on the inference engine, verifies that your server is running via a health check to `localhost:44468`, and then streams weight downloads right into your unified memory for the `90M` or `8B` models.

---

## Exploring the Engine 

Here is what you can do next using these Python scripts:

### 1. The Interactive Shell (Recommended!)
The fastest way to experience the engine is via the Interactive Shell. Built beautifully with `rich` formatting, the shell visually connects you to all backend APIs.

```bash
python interactive_shell.py
```
**Features inside:**
* **Real-Time Continuous Batching Visualizer (Option 5)**: This will prompt you to run multiple heavy queries concurrently. It dynamically formats a real-time tracking table that partitions *thinking* tokens using a dim grey color, and streams *visible replies* synchronously across all workers.
* **Interactive Chat (Option 6)**: Converse with local models like ChatGPT, fully supporting reasoning prefixes. 
* **Model State Management**: Download, load, array, and aggressively unload handler processes dynamically.

### 2. HTTP Concurrency Benchmarks
Curious exactly how Continuous Batching saves resources and scales against load?

```bash
python benchmark_http_concurrency.py
```
This script sweeps the engine at `Concurrency=8`, `Concurrency=16`, and `Concurrency=32` respectively. It compares Time To First Token (TTFT), tokens/sec throughput bounds, and calculates how the model linearly scales across high-traffic HTTP loads against a static endpoint.

### 3. Pre-fill Configurations Sweeper 
How do prefill-chunk limits affect memory?
```bash
python sweep_cb_configs.py
```
This iteratively injects different `cb_max_num_seqs` overrides to load the backend into an optimized configuration and profiles execution throughput on cached vs mixed-query workloads.

### 4. Streaming Performance Profiler
Used for exact tracking of real-world streaming TTFT limitations and inter-token latency bounds over SSE.
```bash
python benchmark_streaming.py
```

### 5. Unified Dashboard
A passive UI to just watch inference logs, model IDs, CPU/GPU usage, memory bandwidth, and process health limits simultaneously.
```bash
python llm_health_dashboard.py
```
