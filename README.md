### What we recommend

The easiest way to get started is by using our interactive setup script. It will configure terminal-based monitoring tool, download your first model, and let you test the engine's capabilities via benchmarks or the interactive chat shell.

```bash
# Make the script executable
chmod +x setup.sh

# Run the interactive setup, honestly this is the best way to get started
./setup.sh
```

---

# Bodega Inference Engine

Bodega Inference Engine delivers enterprise-grade inference directly on your machine. Built specifically for Apple Silicon, it provides a seamless runtime with openai-compatible api which is faster, more memory efficient and intuitive than any other runtime available out there.

**Architecture:** Multi-process isolated handler architecture prevents Metal memory leaks.

As of the latest release, Bodega is a **multi-model registry** — you can load, route to, and unload multiple models simultaneously, each running in its own hardware-isolated subprocess. The engine automatically handles resource allocation and delivers the fastest possible inference on Apple Silicon.

**Key Capabilities:**
- Multi-model registry with dynamic loading and unloading
- Language model inference with streaming support
- Multimodal language model support (vision)
- Image generation (live next week, week of March 17)
- Image editing (live next week, week of March 17)
- Structured output via JSON schema constraints
- Speculative decoding for accelerated generation
- Continuous batching for high-throughput workloads
- Built-in prompt caching

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Endpoints](#core-endpoints)
3. [Model Management](#model-management)
4. [Model Discovery](#model-discovery)
5. [Advanced Features](#advanced-features)
   - [Reasoning Models](#reasoning-models)
   - [JSON Mode](#json-mode)
   - [Prompt Caching](#prompt-caching)
   - [Speculative Decoding](#speculative-decoding)
   - [Continuous Batching](#continuous-batching-high-throughput)
   - [Custom Chat Templates](#custom-chat-templates)
6. [Monitoring and Health](#monitoring-and-health)
7. [Best Practices](#best-practices)

---

## Getting Started

### Quick Implement

Start the server and load your first model:

```bash

# Or dynamically load a model via API
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_id": "bodega-raptor-8b",
    "model_type": "lm",
    "context_length": 32768,
    "prompt_cache_size": 10
  }'

# Make your first inference request
curl -X POST http://localhost:44468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bodega-raptor-8b",
    "messages": [
      {"role": "user", "content": "Hello, welcome to the world of dreamers?"}
    ]
  }'
```

### Quick Start (Multi-Model Registry)

The standard way to run multiple models is to keep calling the `/v1/admin/load-model` endpoint — each call spawns a new isolated subprocess for that model. You can check which models are running and their memory usage at any time via `/health`:

```bash
# Load a language model
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "bodega-raptor-90m",
    "model_type": "lm",
    "model_path": "srswti/bodega-raptor-90m"
  }'

# Load a multimodal model alongside it
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "srswti/bodega-solomon-9b",
    "model_type": "multimodal",
    "model_path": "srswti/bodega-solomon-9b"
  }'

# Load our favourite model alongside it :)
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "blackbird",
    "model_type": "lm",
    "model_path": "srswti/blackbird-she-doesnt-refuse-21b",
    "context_length": 32768,
    "max_concurrency": 1,
    "reasoning_parser": "harmony",
    "tool_call_parser": "harmony"
  }'

# Check what's running
curl http://localhost:44468/health
```

```json
{
  "status": "ok",
  "model_id": "bodega-raptor-0.9b, srswti/bodega-solomon-9b",
  "model_status": "initialized (2 model(s))",
  "models_detail": [
    {
      "id": "bodega-raptor-0.9b",
      "type": "lm",
      "status": "running",
      "ram_usage_mb": 3667.1
    },
    {
      "id": "srswti/bodega-solomon-9b",
      "type": "multimodal",
      "status": "running",
      "ram_usage_mb": 13344.5
    }
  ]
}
```

You can also load any HuggingFace model directly — not just SRSWTI models. For example, loading a community Qwen model with continuous batching:

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "Qwen/Qwen3-30B-A3B-MLX-4bit",
    "model_type": "lm",
    "model_path": "Qwen/Qwen3-30B-A3B-MLX-4bit",
    "max_concurrency": 1,
    "queue_timeout": 300,
    "queue_size": 100,
    "continuous_batching": true,
    "cb_max_num_seqs": 256,
    "cb_prefill_batch_size": 16,
    "cb_completion_batch_size": 32
  }' | python3 -m json.tool
```

```json
{
  "status": "loaded",
  "model_id": "Qwen/Qwen3-30B-A3B-MLX-4bit",
  "model_path": "Qwen/Qwen3-30B-A3B-MLX-4bit",
  "model_type": "lm"
}
```

> **Note:** `config.yaml` support for launching multiple models at server start is currently in experimental release for a limited set of users. General availability coming soon.

**Example `config.yaml`:**
```yaml
server:
  host: "0.0.0.0"
  port: 44468

models:
  - model_id: "bodega-solomon-9b"
    model_type: "multimodal"
    model_path: "srswti/bodega-solomon-9b"
    max_concurrency: 1

  - model_id: "bodega-raptor-8b"
    model_type: "lm"
    model_path: "srswti/bodega-raptor-8b-mxfp4"
    prompt_cache_size: 10
```

### Python Quick Start

```python
import requests

BASE_URL = "http://localhost:44468"

# Load a model
response = requests.post(
    f"{BASE_URL}/v1/admin/load-model",
    json={
        "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
        "model_type": "lm",
        "context_length": 32768
    }
)
print(response.json())

# Chat completion
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "bodega-raptor-8b",
        "messages": [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

---

## Core Endpoints

### Chat Completions

Generate text responses using loaded language models. Fully compatible with OpenAI's chat completions API.

**Endpoint:** `POST /v1/chat/completions`

#### Basic Request

```bash
curl -X POST http://localhost:44468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bodega-raptor-8b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

#### Streaming Response

```bash
curl -X POST http://localhost:44468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bodega-raptor-8b",
    "messages": [
      {"role": "user", "content": "Write a short story about AI."}
    ],
    "stream": true
  }'
```

#### Python Streaming Example

```python
import requests
import json

response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "bodega-raptor-8b",
        "messages": [
            {"role": "user", "content": "Write a short story about AI."}
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data != '[DONE]':
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    print(content, end="", flush=True)
```

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model identifier |
| `messages` | array | required | Array of message objects with role and content |
| `max_tokens` | integer | null | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0.0 to 2.0) |
| `top_p` | float | 1.0 | Nucleus sampling parameter |
| `stream` | boolean | false | Enable streaming responses |
| `tools` | array | null | Available tools for function calling |
| `tool_choice` | string/object | "auto" | Control tool selection behavior |
| `response_format` | object | null | Specify output format (e.g., JSON schema) |
| `presence_penalty` | float | 0.0 | Penalize new tokens based on presence |
| `frequency_penalty` | float | 0.0 | Penalize new tokens based on frequency |
| `stop` | string/array | null | Stop sequences |
| `seed` | integer | null | Random seed for reproducibility |

#### Response Format

```json
{
  "id": "chatcmpl_1234567890",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "bodega-raptor-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 120,
    "total_tokens": 135
  }
}
```

---

### Structured Outputs (JSON Schema)

Force the model to output data that strictly adheres to a predefined JSON schema. Constraints are applied natively within the inference engine using outlines.

**Endpoint:** `POST /v1/chat/completions`

```python
import requests

schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "AddressExtractor",
        "schema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string", "description": "2 letter abbreviation"},
                        "zip": {"type": "string", "description": "5 digit zip code"}
                    },
                    "required": ["street", "city", "state", "zip"]
                }
            },
            "required": ["address"]
        }
    }
}

response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "bodega-raptor-8b",
        "messages": [
            {"role": "system", "content": "Extract the address from the user input into the specified JSON format."},
            {"role": "user", "content": "Please format this address: 1 Hacker Wy Menlo Park CA 94025"}
        ],
        "response_format": schema,
        "stream": False
    }
)

# Returns: '{"address": {"street": "1 Hacker Wy", "city": "Menlo Park", "state": "CA", "zip": "94025"}}'
print(response.json()["choices"][0]["message"]["content"])
```

Structured output also works with `"stream": true` — the model will stream partial JSON tokens as they are generated.

---

### Multimodal Completions (Vision)

Pass images alongside text prompts for models with vision capabilities such as `bodega-solomon-9b`.

**Endpoint:** `POST /v1/chat/completions`

#### URL Image

```bash
curl -X POST http://localhost:44468/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ 
    "model": "srswti/bodega-solomon-9b", 
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image? Provide a detailed description."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://weblog.spots.ag/08-2018/aventador_svj_official/th.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'

```

#### Local Base64 Image

```python
import base64
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("document_scan.png")

response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "bodega-solomon-9b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this scanned document."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ]
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

---

### Image Generation

Generate images from text prompts using locally-running image models.

> **Coming week of March 17.** 
Right now its a experimental release

**Endpoint:** `POST /v1/images/generations`

First, load an image generation model using one of the available `config_name` values:

```bash
# Solomon — fast, lightweight generation (recommended starting point)
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "solomon",
    "model_type": "image-generation",
    "config_name": "solomon"
  }'

# Keshav — turbo generation, extremely fast
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "keshav",
    "model_type": "image-generation",
    "config_name": "keshav"
  }'
```

Then generate:

```bash
curl -X POST http://localhost:44468/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "solomon",
    "prompt": "A highly detailed portrait of a tiny red dragon wearing a chef hat, pulling a fresh loaf of sourdough bread out of a medieval stone oven.",
    "size": "1024x1024",
    "guidance_scale": 3.5,
    "steps": 14,
    "seed": 42
  }'
```

Available `config_name` values for image generation: `solomon`, `solomon-max`, `rehoboam`, `omri-4b`, `omri-9b`, `keshav`, `kalamkari`, `fibo`.

The engine returns a standard OpenAI-compatible object with a `b64_json` image payload:

```json
{
  "created": 1709428581,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

---

### Image Editing

Edit existing images with text instructions using `srswti/keshav` or `srswti/kalamkari`.

> **Coming week of March 17.**

**Endpoint:** `POST /v1/images/edits`

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "kalamkari-edit",
    "model_type": "image-edit",
    "config_name": "qwen-image-edit"
  }'
```


## Model Management

Bodega is a multi-model registry. You can dynamically spawn, route to, and unload process-isolated model handlers without ever restarting the server.

### Load Model

Spawn a new handler process for a model. It becomes immediately available for inference requests.

**Endpoint:** `POST /v1/admin/load-model`

#### Load a Language Model

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_id": "bodega-raptor-8b",
    "model_type": "lm",
    "context_length": 32768,
    "max_concurrency": 1,
    "prompt_cache_size": 10
  }'
```

#### Load an Image Generation Model

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "srswti/solomon",
    "model_type": "image-generation",
    "config_name": "solomon",
    "quantize": 8
  }'
```

#### Python Example

```python
import requests

response = requests.post(
    "http://localhost:44468/v1/admin/load-model",
    json={
        "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
        "model_type": "lm",
        "context_length": 32768,
        "max_concurrency": 1,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3"
    }
)
print(response.json())
```

#### Mapping HuggingFace Model Types to `model_type`

When loading any model from HuggingFace — not just SRSWTI models — use the HuggingFace model card to determine the right `model_type`. The two most common cases:

**`text-generation` on HuggingFace → `model_type: "lm"`**

These are standard language models that take text in and produce text out. Any model whose HuggingFace page lists the pipeline tag as `text-generation` should be loaded with `"lm"`.

```bash
# Example: a community Qwen text generation model
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3-8b",
    "model_type": "lm",
    "model_path": "mlx-community/Qwen3-8B-4bit",
    "context_length": 32768
  }'
```

**`image-text-to-text` on HuggingFace → `model_type: "multimodal"`**

These are vision-language models that accept both images and text as input. Any model whose HuggingFace page lists the pipeline tag as `image-text-to-text` should be loaded with `"multimodal"`. This applies to models like Qwen-VL, LLaVA, InternVL, and others.

```bash
# Example: a community vision model
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3.5-27b-vl",
    "model_type": "multimodal",
    "model_path": "mlx-community/Qwen3.5-27B-4bit",
    "context_length": 16384
  }'
```

Once loaded as `multimodal`, you can pass images in the standard `image_url` content block format just like with `bodega-solomon-9b`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | required | HuggingFace repo ID or local path |
| `model_id` | string | null | Alias used in API requests (defaults to path) |
| `model_type` | string | "lm" | Model type: `lm`, `multimodal`, `image-generation`, `image-edit` |
| `context_length` | integer | 32768 | Maximum context length |
| `max_concurrency` | integer | 1 | Maximum concurrent requests |
| `queue_timeout` | integer | 300 | Request timeout in seconds |
| `queue_size` | integer | 100 | Maximum queue size |
| `quantize` | integer | 8 | Quantization level for Flux models (4, 8, or 16) |
| `config_name` | string | null | Config for image generation: `solomon`, `solomon-max`, `rehoboam`, `omri-4b`, `omri-9b`, `keshav`, `kalamkari`, `fibo`. For editing: `flux-kontext-dev`, `flux2-klein-edit-4b`, `flux2-klein-edit-9b`, `qwen-image-edit` |
| `lora_paths` | array | null | Paths to LoRA adapters |
| `lora_scales` | array | null | Scale factors for LoRA adapters |
| `disable_auto_resize` | boolean | false | Disable auto-resize for vision models |
| `enable_auto_tool_choice` | boolean | false | Enable automatic tool selection |
| `tool_call_parser` | string | null | Parser for tool calls (`qwen3`, `harmony`, etc.) |
| `reasoning_parser` | string | null | Parser for reasoning content (`qwen3`, `harmony`, etc.) |
| `trust_remote_code` | boolean | false | Allow custom model code execution |
| `chat_template_file` | string | null | Path to custom chat template |
| `continuous_batching` | boolean | false | Enable high-throughput continuous batching |
| `cb_max_num_seqs` | integer | 256 | Max sequences in the batching engine |
| `cb_prefill_batch_size` | integer | 8 | Concurrency limit for prompt ingestion |
| `cb_completion_batch_size` | integer | 32 | Generation concurrency limit on GPU |
| `cb_chunked_prefill_tokens` | integer | 2048 | Token chunk size for large prompts |
| `cb_enable_prefix_cache` | boolean | true | Enable block-aware prompt caching |
| `draft_model_path` | string | null | Path to draft model for speculative decoding |
| `num_draft_tokens` | integer | null | Number of tokens for the draft model to guess |
| `prompt_cache_size` | integer | 0 | Number of prompt cache slots |

#### Available Parsers

Both `tool_call_parser` and `reasoning_parser` support: `qwen3`, `glm4_moe`, `qwen3_coder`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `harmony`, `minimax_m2`.

---

### Unload Model

Gracefully shut down a model's subprocess and unregister it from the engine, instantly freeing its unified GPU/CPU memory. The rest of your loaded models continue running uninterrupted.

**Endpoint:** `DELETE /v1/admin/unload-model/{model_id}`

```bash
# Unload by model_id
curl -X DELETE http://localhost:44468/v1/admin/unload-model/bodega-raptor-0.9b

# Works with full path model IDs too
curl -X DELETE http://localhost:44468/v1/admin/unload-model/srswti/bodega-raptor-90m
```

```python
response = requests.delete("http://localhost:44468/v1/admin/unload-model/bodega-raptor-0.9b")
print(response.json())
```

---

### Delete Model

Remove a model from your local HuggingFace cache to free disk space. The model must be unloaded first if it is currently running.

**Endpoint:** `DELETE /v1/models/{model_id}`

```bash
# Delete a locally cached model
curl -X DELETE "http://localhost:44468/v1/models/local/mlx-community/Qwen3.5-27B-4bit"
```

```json
{"id": "mlx-community/Qwen3.5-27B-4bit", "object": "model", "deleted": true}
```

```python
model_id = "SRSWTI/bodega-raptor-8b-mxfp4"
response = requests.delete(f"http://localhost:44468/v1/models/local/{model_id}")
print(response.json())
```

---

### List Loaded Models & Memory Usage

Retrieve real-time Metal Unified Memory and CPU RSS metrics for all running models.

**Endpoint:** `GET /v1/admin/loaded-models`

```bash
curl http://localhost:44468/v1/admin/loaded-models
```

```python
response = requests.get("http://localhost:44468/v1/admin/loaded-models")
models = response.json().get("data", [])

for model in models:
    print(f"[{model['status'].upper()}] {model['id']} — PID: {model['pid']}")
    mem = model.get('memory', {})
    print(f"  └ Metal Active (GPU): {mem.get('metal_active_mb', 0):.1f} MB")
    print(f"  └ Process RSS overhead (CPU): {mem.get('rss_mb', 0):.1f} MB")
    print(f"  └ Total System Pool: {mem.get('total_mb', 0):.1f} MB\n")
```

**Response Format:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "bodega-raptor-8b",
      "type": "lm",
      "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
      "context_length": 32768,
      "created_at": 1704067200,
      "status": "running",
      "pid": 83932,
      "memory": {
        "metal_active_mb": 4150.2,
        "metal_cache_mb": 0.0,
        "metal_peak_mb": 4150.2,
        "rss_mb": 408.2,
        "total_mb": 4558.4
      }
    }
  ],
  "total": 1
}
```

---

## Model Discovery

Discover, download, and manage models from HuggingFace.

### List Available Models

List all models in your local HuggingFace cache.

**Endpoint:** `GET /v1/models`

```bash
curl http://localhost:44468/v1/models

# Verify download completeness against HuggingFace API
curl "http://localhost:44468/v1/models?verify_with_hub=true"
```

**Response Format:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "SRSWTI/bodega-raptor-8b-mxfp4",
      "object": "model",
      "created": 1704067200,
      "owned_by": "SRSWTI",
      "size_gb": 4.8,
      "download_percentage": 100.0,
      "is_complete": true
    }
  ]
}
```

---

### Download Model

Download a model to your local cache.

**Endpoint:** `POST /v1/admin/download-model`

```bash
curl -X POST http://localhost:44468/v1/admin/download-model \
  -H "Content-Type: application/json" \
  -d '{"model_path": "SRSWTI/bodega-raptor-8b-mxfp4"}'
```

---

### Download Model with Progress

Download with real-time progress via Server-Sent Events.

**Endpoint:** `POST /v1/admin/download-model-stream`

```python
import requests, json

response = requests.post(
    "http://localhost:44468/v1/admin/download-model-stream",
    json={"model_path": "SRSWTI/bodega-raptor-8b-mxfp4"},
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: ') and line[6:] != '[DONE]':
            progress = json.loads(line[6:])
            print(f"{progress['status']} — {progress.get('progress', 0)}%")
            if 'current_file' in progress:
                print(f"  File: {progress['current_file']}")
```

---

## Advanced Features

### Reasoning Models

Some models support an explicit reasoning/thinking process. Configure a parser to extract it.

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_type": "lm",
    "reasoning_parser": "qwen3"
  }'
```

```python
response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "bodega-raptor-8b",
        "messages": [{"role": "user", "content": "Solve this logic puzzle: ..."}],
        "chat_template_kwargs": {"enable_thinking": True}
    }
)

message = response.json()["choices"][0]["message"]

if "reasoning_content" in message:
    print("Thinking:", message["reasoning_content"])

print("Answer:", message["content"])
```

---

### JSON Mode

Force the model to output valid JSON.

```python
response = requests.post(
    "http://localhost:44468/v1/chat/completions",
    json={
        "model": "bodega-raptor-8b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {"role": "user", "content": "List three colors with their hex codes."}
        ],
        "response_format": {"type": "json_object"}
    }
)

import json
result = json.loads(response.json()["choices"][0]["message"]["content"])
print(result)
```

---

### Prompt Caching

Bodega uses dynamic prompt caching for extremely fast time-to-first-token on recurring sequences. The cache operates natively on MLX token indices — overlapping prefixes across subsequent calls bypass matrix multiplication completely.

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_type": "lm",
    "prompt_cache_size": 25
  }'
```

---

### Speculative Decoding

Speculative decoding significantly accelerates generation for large models — especially in single-user, latency-sensitive workloads — without any change to output quality or the response format you receive.

#### Why generation is slow on large models

On Apple Silicon, text generation is **memory-bandwidth-bound**, not compute-bound. For every single token a large model generates, the GPU must load the full set of model weights from unified memory into the compute cores. A 8B parameter model at 4-bit quantization is roughly 4–5GB. Loading those weights once to produce a single token means the vast majority of each generation step is spent on memory transfer, not math. This is why scaling up GPU cores doesn't help much — you're waiting on the memory bus, not the ALUs.

#### What speculative decoding does

Instead of running the large target model once per token, the engine runs two models in parallel:

1. **Draft model** — a small, fast model (e.g. 0.6B params) that guesses the next `N` tokens very quickly. Because it's tiny, this costs almost nothing.
2. **Target model** — the large model you actually want responses from. Instead of generating one token at a time, it evaluates all `N` draft guesses in a single forward pass using parallel matrix multiplication.

If the target model agrees with the draft's guesses, all `N` tokens are accepted at once. You get `N` tokens for the memory-load cost of one. When the target disagrees at position `k`, it accepts tokens 0 through `k-1` and corrects at `k`, and the draft restarts from there.

In practice, a well-matched draft model (same tokenizer family, same training distribution) agrees on the majority of guesses, yielding effective speedups of 2–3x on generation-heavy workloads without touching output quality. The output is mathematically identical to what the target model would have generated on its own.

#### Requirements

The draft model **must share the same tokenizer** as the target model. Using a model from a different family (e.g. a Llama draft with a Qwen target) will produce garbage. Use a smaller variant from the same model family — for example, a `0.6B` or `1B` Qwen3 variant to accelerate a `8B` or `32B` Qwen3 target.

> **Note:** Speculative decoding and continuous batching cannot be used simultaneously. Speculative decoding is optimal for single-user latency. Continuous batching is optimal for multi-user throughput. Choose based on your workload.

#### Configuration

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_type": "lm",
    "draft_model_path": "Qwen/Qwen3-0.6B-MLX-8bit",
    "num_draft_tokens": 4
  }'
```

Or via `config.yaml` (experimental):

```yaml
models:
  - model_id: "raptor-fast"
    model_type: "lm"
    model_path: "srswti/bodega-raptor-8b-mxfp4"
    draft_model_path: "Qwen/Qwen3-0.6B-MLX-8bit"
    num_draft_tokens: 3
```

#### Response

The response format is identical to a standard completion — no extra fields, no proprietary metrics. The only observable difference is that the payload arrives faster. The `completion_tokens` count reflects what the target model produced, not the draft speculation.

```json
{
  "id": "chatcmpl_2fa419e...",
  "object": "chat.completion",
  "model": "raptor-fast",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Here's your answer...",
        "role": "assistant"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 21,
    "total_tokens": 121,
    "completion_tokens": 100,
    "prompt_tokens_details": {
      "cached_tokens": 3
    }
  }
}
```

---

### Continuous Batching (High Throughput)

Bodega's continuous batching engine maximizes throughput for multi-user workloads on Apple Silicon. It is the primary mechanism for serving multiple concurrent users efficiently, and the numbers are dramatic — small SRSWTI models and community models like `mlx-community/Qwen3.5-2B-6bit` approach **~900 tok/s system throughput** on an m1 Max when measured in-process. At the HTTP server layer, measured throughput currently reaches **~600 tok/s** — the gap is not the inference engine, it is the HTTP serialization layer, and we are actively working to close it. See the [HTTP Bottleneck](#the-http-bottleneck) section below for details.

#### How It Works

**The Continuous Batching Flow:**
1. Request A arrives. The engine processes A's prompt and starts generating token 1.
2. Request B arrives. Instead of waiting, the engine's Scheduler **injects B into the active batch instantly**.
3. On the very next step, the GPU processes **both** A's token generation AND B's prompt processing simultaneously.
4. The output is streamed back dynamically: token 2 for A, and token 1 for B.
5. If Request A hits a stop word and finishes, it is **ejected from the batch immediately**, freeing up space for Request C, while Request B simply continues generating.

**Why this is blazingly fast:**
Because Apple Silicon is bottlenecked by memory bandwidth during text generation, fetching the model weights accounts for roughly 80% of the time. If you can fetch the weights *once* and use them to multiply against *four* different requests simultaneously, you get nearly 4x the throughput with almost zero latency penalty.

This is called "continuous" because requests enter and exit the active GPU batch fluidly as they arrive and finish, without waiting for the whole batch to complete.

#### Sequential vs. Continuous Batching

The difference is most visible in TTFT (time to first token) under concurrent load. In sequential mode, request 8 waits for requests 1–7 to finish — TTFT grows linearly with queue depth. In continuous batching, all requests are injected into the active batch and begin generating almost immediately.

Benchmarked on the **blackbird-she-doesnt-refuse-21b** model on M1 MAX 64gb:

| Concurrency | Sequential Mean TTFT | CB Mean TTFT | Sequential Throughput | CB Throughput |
|-------------|----------------------|--------------|-----------------------|---------------|
| 4 | 6,510ms | **541ms** | 44.4 tok/s | 37.7 tok/s |
| 8 | 12,837ms | **247ms** | 44.1 tok/s | 49.2 tok/s |

At concurrency 8, continuous batching delivers a **52x improvement in TTFT** — from 12.8 seconds to 247ms. Sequential throughput is flat because it's bottlenecked by single-request speed. CB throughput scales by saturating GPU parallelism across concurrent sequences.

#### Configuration Examples

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_type": "lm",
    "continuous_batching": true,
    "cb_max_num_seqs": 256,
    "cb_prefill_batch_size": 16,
    "cb_completion_batch_size": 32
  }'
```

Or via `config.yaml` (experimental):

```yaml
models:
  - model_id: "raptor-batched"
    model_type: "lm"
    model_path: "srswti/bodega-raptor-8b-mxfp4"
    continuous_batching: true
    cb_max_num_seqs: 256
    cb_prefill_batch_size: 16
    cb_completion_batch_size: 32
```

#### The Configuration Flags Explained

To tune the batching engine, you have 5 main levers:

##### 1. `--cb-max-num-seqs` (Default: 256)
*What it is:* The absolute maximum number of sequences (requests) the engine is allowed to hold in its scheduler at one time.
*How to tune:*
- If this is too low, requests will be rejected under heavy load.
- If it's too high, you might run out of KV-cache memory, causing MLX to swap to disk (very slow).
- Set this based on your available RAM. 256 is safe for M1/M2/M3 Max chips (64GB) with 8B models.

##### 2. `--cb-completion-batch-size` (Default: 32)
*What it is:* The maximum number of sequences that can be actively *generating tokens* in the GPU simultaneously.
*How to tune:*
- Above ~32 concurrent generations, you start hitting computation limits on Apple Silicon GPUs, and individual Time-To-First-Token (TTFT) or Time-Per-Output-Token (TPOT) will rise.
- 32 means MLX will multiply the weights against a matrix of size 32 on every generation step.

##### 3. `--cb-prefill-batch-size` (Default: 8)
*What it is:* When a burst of 50 new requests arrives, how many of them do we inject into the active batch on the *very next step*?
*How to tune:*
- Prefilling (processing the initial prompt) is computationally heavy. If you try to prefill 50 prompts at once, the GPU hangs for several seconds. If there are other requests currently generating tokens, those users will experience a massive stutter.
- By capping this at 8, we ensure that new requests are digested in small bites. The active generation stream might pause for 100ms instead of 3000ms.

##### 4. `--cb-chunked-prefill-tokens` (Default: 2048)
*What it is:* What if a single user submits a massive 16,000-token prompt? That alone will block the GPU. Chunked prefill solves this by splitting that 16K prompt into 2048-token chunks.
*How to tune:*
- During step 1, it processes chunk 1 (0-2048) alongside the active token generations.
- Step 2: chunk 2 (2048-4096) + active generations.
- This entirely eliminates the "long prompt stutter" problem for concurrent users. Set to 0 to disable.

##### 5. `--cb-enable-prefix-cache` (Default: True)
*What it is:* Automatic prompt caching. If User A asks a question about a 10,000 token document, the engine calculates the KV-cache and stores it in memory blocks. If User B asks a different question about *the exact same document*, the engine recognizes the shared prefix and instantly reuses the 10,000 token cache, dropping TTFT from seconds to milliseconds.
*How to tune:* Leave it on. It uses block-aware memory management to automatically evict the oldest prefixes when you hit MLX memory pressure.

So here are the Tuning Parameters

| Parameter | Recommended | What It Controls |
|-----------|-------------|-----------------|
| `cb_max_num_seqs` | 256 | Total scheduler capacity — active + waiting sequences combined. Lower this to 64 on 16GB Macs with large models to prevent KV-cache overflow and disk swapping. |
| `cb_completion_batch_size` | 32 | Max concurrent token generations per GPU step. The primary throughput lever. Above ~32 on small models, Apple Silicon hits compute saturation and per-token speed degrades. For 21B+ models, cap at 16. |
| `cb_prefill_batch_size` | 8–16 | How many new prompt-ingestion requests are allowed to enter the active batch per step. This is your TTFT fairness lever. Higher values process bursts faster but can cause brief generation stutter for active streams during the prefill phase. |
| `cb_chunked_prefill_tokens` | 2048 | Splits very long prompts into chunks ingested across multiple steps. Prevents a single massive-context request from freezing generation for everyone else. |
| `cb_enable_prefix_cache` | true | Block-aware KV-cache. Recognizes shared prefixes across requests (identical system prompts, shared documents) and reuses computed KV blocks, eliminating re-ingestion entirely. |

#### Benchmark Results


**blackbird-she-doesnt-refuse-21b (Hybrid SWA/Global attention)**

| Concurrency | Wall Time | Throughput | Mean TTFT | P95 TTFT |
|-------------|-----------|------------|-----------|----------|
| 4 | 22.22s | 37.7 tok/s | 541ms | 1463ms |
| 8 | 13.96s | 49.2 tok/s | 247ms | 372ms |
| 16 | 16.71s | **63.7 tok/s** | 1444ms | 2880ms |

Peak: **1.69x** throughput gain. Gains plateau after concurrency 8 — this model is memory-bandwidth-bound at 21B. TTFT climbs at concurrency 16 as the prefill queue builds up.

---

**deepseek-raptor-32b-4bit**

| Concurrency | Wall Time | Throughput | Mean TTFT | P95 TTFT |
|-------------|-----------|------------|-----------|----------|
| 1 | 188.25s | 8.8 tok/s | 461ms | 1629ms |
| 4 | 161.37s | 9.3 tok/s | 773ms | 1356ms |
| 8 | 145.56s | **10.4 tok/s** | 9,802ms | 36,049ms |
| 16 | 162.22s | 9.8 tok/s | 39,025ms | 93,014ms |

Peak: **1.18x** gain, marginal. At 32B, this model is heavily compute-bound. Adding batch concurrency provides minimal throughput benefit while TTFT explodes. **Recommended concurrency: 1–4.**

---

**bodega-raptor-0.9b**

| Concurrency | Wall Time | Throughput | Mean TTFT | P95 TTFT |
|-------------|-----------|------------|-----------|----------|
| 4 | 46.26s | 57.3 tok/s | 234ms | 976ms |
| 8 | 44.41s | 58.5 tok/s | 241ms | 318ms |
| 16 | 32.30s | 80.7 tok/s | 440ms | 567ms |
| 32 | 20.57s | **127.5 tok/s** | 923ms | 1455ms |

Peak: **2.23x** throughput gain at concurrency 32. Strong scaling characteristic of sub-1B models where the GPU is purely bandwidth-bound.

---

**mlx-community/Qwen3.5-2B-6bit**

| Concurrency | Wall Time | Throughput | Mean TTFT | P95 TTFT |
|-------------|-----------|------------|-----------|----------|
| 4 | 12.96s | 313.3 tok/s | 204ms | 1239ms |
| 8 | 10.13s | 384.2 tok/s | 126ms | 255ms |
| 16 | 9.22s | 447.2 tok/s | 290ms | 458ms |
| 32 | 6.87s | **619.8 tok/s** | 596ms | 617ms |

Peak: **1.98x** gain at ~620 tok/s measured over HTTP. The ~900 tok/s figure was measured by calling the batching engine directly in-process — no HTTP server, no SSE serialization, no network stack. That number represents the raw inference ceiling on m4 Max. The ~280 tok/s gap you see in the HTTP benchmark is entirely the server layer, not the inference engine. See [The HTTP Bottleneck](#the-http-bottleneck) below.

---

**Detailed sweep — mlx-community/Qwen3.5-2B-6bit (mixed and same-query)**

| Scenario | Concurrency | Prefill Batch | Mean TTFT | P95 TTFT | Per-Req TPS | System Throughput |
|----------|-------------|---------------|-----------|----------|-------------|-------------------|
| Mixed | 8 | 2 | 196ms | 316ms | 58.7 | 450.5 tok/s |
| Mixed | 8 | 4 | 206ms | 289ms | 59.9 | 451.6 tok/s |
| Mixed | 8 | 8 | 258ms | 259ms | 62.4 | **462.6 tok/s** |
| Mixed | 16 | 4 | 344ms | 557ms | 36.0 | 534.0 tok/s |
| Mixed | 16 | 8 | 321ms | 425ms | 36.6 | 536.0 tok/s |
| Mixed | 16 | 16 | 331ms | 332ms | 37.3 | **547.4 tok/s** |
| Mixed | 32 | 8 | 384ms | 676ms | 31.2 | 889.5 tok/s |
| Mixed | 32 | 16 | 424ms | 616ms | 31.4 | **901.3 tok/s** |
| Same query | 8 | 2 | 165ms | 278ms | 60.7 | 439.7 tok/s |
| Same query | 8 | 4 | **144ms** | 198ms | 64.0 | 475.8 tok/s |
| Same query | 8 | 8 | 162ms | **162ms** | 64.5 | **480.1 tok/s** |
| Same query | 16 | 4 | 316ms | 543ms | 35.8 | 517.9 tok/s |
| Same query | 16 | 8 | 269ms | 365ms | 36.8 | 543.8 tok/s |
| Same query | 16 | 16 | 301ms | 302ms | 37.6 | **556.2 tok/s** |
| Same query | 32 | 8 | 469ms | 773ms | 31.0 | 870.3 tok/s |
| Same query | 32 | 16 | 467ms | 638ms | 31.8 | **902.2 tok/s** |

#### Key Takeaways

**1. Throughput scales near-linearly with concurrency for small models.** Without CB, system throughput equals per-request TPS (~60 tok/s). With CB at concurrency 32, you reach ~900 tok/s system throughput — a **15x total throughput gain** on the same hardware.

**2. Prefill batch size is a TTFT fairness lever, not a throughput lever.** Notice P95 TTFT at concurrency 16: with prefill batch 4, P95 is 557ms — some users are waiting because they're stuck behind multiple prefill rounds. With prefill batch 16, P95 drops to 332ms and mean is 331ms — everyone in the burst gets their first token at nearly the same time. The rule: if you expect burst traffic (many requests arriving simultaneously), set a higher prefill batch. If requests arrive organically over time, a lower prefill batch keeps active generation streams smoother.

**3. Prefix caching is a meaningful TTFT accelerator.** At concurrency 8, mixed queries average 196–258ms TTFT. The same query (shared prefix, cache hit for all subsequent requests) drops to **144ms** mean TTFT with a P95 of 198ms. The engine computed the prompt KV-cache once and reused it across all 8 requests. Per-request TPS also climbs from ~59 to ~64.5 because subsequent requests skip prompt ingestion entirely.

**4. Large models (21B+) have a concurrency sweet spot.** For the 32B model, optimal concurrency is 1–4. Pushing to 8+ concurrent requests causes TTFT to spike into the tens of seconds — the GPU is compute-saturated and the KV-cache grows large enough to risk swapping. For 21B models, concurrency 8 is the practical ceiling before TTFT becomes unacceptable for real-time users.

---

#### The HTTP Bottleneck

For small and mid-size models, the batching engine is fast enough that the HTTP server itself becomes the bottleneck — a situation that is uncommon in most inference systems and speaks to how aggressively the Bodega inference engine saturates Apple Silicon's memory bandwidth.

**What's happening:** When the batching engine generates tokens, it produces them in steps. Each step generates one token per active sequence simultaneously, then the output needs to be: serialized to JSON, wrapped in a Server-Sent Events `data:` frame, written to each open HTTP response stream, and flushed through the OS network stack. For large models generating at 8–30 tok/s, this overhead is negligible. For a model running at 900 tok/s in-process across 32 concurrent streams, each engine step completes in milliseconds — and the HTTP layer starts struggling to keep up with the token emission rate.

The measured gap on M4 Max with `Qwen3.5-2B-6bit`:

| Mode | Throughput |
|------|------------|
| In-process (direct engine call, no HTTP) | ~900 tok/s |
| HTTP with streaming (`text/event-stream`) | ~600 tok/s |
| Gap | ~300 tok/s (~33% overhead) |

The ~300 tok/s gap is not lost inference work — the GPU is generating tokens at the same rate regardless. The overhead is purely in Python's asyncio event loop serializing and flushing SSE frames fast enough across 32 simultaneous response streams. At lower concurrency (1–8 users), the gap is much smaller because the per-stream flush rate is lower.

**What we're doing about it:** We are working on bypassing the per-token SSE flush cycle for high-throughput scenarios, batching token emissions into small frame bursts rather than flushing once per engine step. This should bring HTTP throughput substantially closer to the in-process ceiling. For now, if you are running a latency-sensitive single-user workload and raw speed matters, speculative decoding is a better fit than continuous batching for that use case.

#### Hardware Guidelines

**Small models (90M–8B) on any Mac with 16GB+ RAM:**
- `cb_max_num_seqs`: 256
- `cb_completion_batch_size`: 32
- `cb_prefill_batch_size`: 16

**Large models (14B–32B) on 32GB+ RAM:**
- `cb_max_num_seqs`: 64
- `cb_completion_batch_size`: 16
- `cb_prefill_batch_size`: 4–8

---

### Custom Chat Templates

Override a model's default chat template:

```bash
curl -X POST http://localhost:44468/v1/admin/load-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "SRSWTI/bodega-raptor-8b-mxfp4",
    "model_type": "lm",
    "chat_template_file": "/path/to/custom_template.jinja"
  }'
```

---

## Monitoring and Health

### Health Check

**Endpoint:** `GET /health`

```bash
curl http://localhost:44468/health
```

**Healthy (multi-model):**
```json
{
  "status": "ok",
  "model_id": "bodega-solomon-9b, bodega-raptor-8b",
  "model_status": "initialized (2 model(s))",
  "models_detail": [
    {"id": "bodega-solomon-9b", "type": "multimodal", "status": "running", "ram_usage_mb": 11645.8},
    {"id": "bodega-raptor-8b", "type": "lm", "status": "running", "ram_usage_mb": 4558.4}
  ]
}
```

**No models loaded:**
```json
{
  "status": "unhealthy",
  "model_id": null,
  "model_status": "no_models"
}
```

---

### Queue Statistics

**Endpoint:** `GET /v1/queue/stats`

```bash
curl http://localhost:44468/v1/queue/stats
```

```python
response = requests.get("http://localhost:44468/v1/queue/stats")
stats = response.json()["queue_stats"]
print(f"Queue size: {stats.get('queue_size', 0)}")
print(f"Active requests: {stats.get('active_requests', 0)}")
```

---

## Best Practices

### Model Selection

---

**Our Open source Work**
- **Explore our Models:** [Hugging Face](https://huggingface.co/srswti)
- **Coding CLI:** [axe on GitHub](https://github.com/SRSWTI/axe)

**Fastest (edge/laptop):**
- `SRSWTI/bodega-raptor-90m` — Sub-100M params, exceptional tool calling and reasoning at the edgehttps://huggingface.co/srswti
- `SRSWTI/bodega-raptor-0.9b` — 400+ tok/s, ideal for classification and query reformulation
- `SRSWTI/axe-turbo-1b` — Sub-50ms first token, edge-first agentic coding

**Balanced performance:**
- `SRSWTI/bodega-raptor-1b-reasoning-opus4.5-distill` — Distilled from Claude Opus 4.5 reasoning patterns
- `SRSWTI/bodega-vertex-4b` — Optimized for structured data processing
- `SRSWTI/bodega-raptor-8b-mxfp4` — Best general-purpose choice for laptops

**Multimodal and agentic:**
- `SRSWTI/bodega-solomon-9b` — Vision + best-in-class agentic coding workflows

**High capacity:**
- `SRSWTI/bodega-raptor-15b-6bit` — Enhanced Raptor variant
- `SRSWTI/bodega-centenario-21b-mxfp4` — Production workhorse, 21B params optimized for sustained workloads
- `SRSWTI/blackbird-she-doesnt-refuse-21b` — Uncensored 21B for unrestricted generation
- `SRSWTI/axe-turbo-31b` — High-capacity desktop/server variant with agentic coding focus

**Flagship intelligence:**
- `SRSWTI/deepseek-v3.2-speciale-distilled-raptor-32b-4bit` — DeepSeek V3.2 distilled to 32B with Raptor reasoning. Exceptional math and code generation in a 5–7GB footprint. 120 tok/s on m4 Max.

---

### Memory Management

- Use the smallest context length that fits your use case
- Unload models you're not actively using to free unified memory
- Monitor queue stats to avoid overloading the scheduler
- Prefer quantized (4-bit or 8-bit) models for better memory efficiency

### Performance Optimization

- Set `max_concurrency: 1` for single-user scenarios
- Use streaming for long responses to improve perceived latency
- Enable `prompt_cache_size` for workloads with recurring prefixes
- Use speculative decoding for single-user, latency-sensitive workloads
- Use continuous batching for multi-user, throughput-sensitive workloads

### Error Handling

```python
response = requests.post("http://localhost:44468/v1/chat/completions", json={...})

if response.status_code == 503:
    print("No model loaded. Load a model first.")
elif response.status_code == 400:
    print("Invalid request parameters.")
elif response.status_code == 200:
    result = response.json()
else:
    print(f"Error: {response.status_code}")
```
### Document Indexing (RAG)

Bodega includes a fully self-contained RAG pipeline for PDF documents

#### Upload & Index a PDF

**Endpoint:** `POST /v1/rag/upload`

```bash
curl -X POST http://localhost:44468/v1/rag/upload \
  -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{
  "file_id": "rag-c6cd8f10",
  "filename": "document.pdf",
  "num_chunks": 71,
  "status": "indexed"
}
```

#### Query an Indexed PDF

The engine embeds your question, retrieves the most relevant chunks via FAISS cosine-similarity, and passes the context alongside your query to the active chat model.

**Endpoint:** `POST /v1/rag/query`

```bash
curl -X POST http://localhost:44468/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "rag-c6cd8f10",
    "query": "What is the main conclusion of this document?",
    "model": "bodega-raptor-8b",
    "top_k": 5
  }'
```

Add `"stream": true` to receive the answer as a Server-Sent Events stream, identical to the standard `/v1/chat/completions` endpoint.

#### List Indexed Documents

**Endpoint:** `GET /v1/rag/documents`

```bash
curl http://localhost:44468/v1/rag/documents
```

#### Delete an Indexed Document

**Endpoint:** `DELETE /v1/rag/documents/{file_id}`

```bash
curl -X DELETE http://localhost:44468/v1/rag/documents/rag-c6cd8f10
```

---
### Security

- The server runs on `localhost:44468` only and is not accessible from external networks
- No authentication is required for local access
- Do not expose this port to the internet without adding proper security measures
- Only set `trust_remote_code: true` for models from verified sources

---

*Documentation last updated: March 2026*
