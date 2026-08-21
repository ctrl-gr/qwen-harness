# Qwen Harness

A small CLI-first agent harness for running `qwen3.5:4b` through a local Ollama server.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Make sure Ollama is running and the model is installed:

```powershell
ollama pull qwen3.5:4b
ollama create qwen3.5:4b-cpu -f .\Modelfile.cpu
```

## Run

```powershell
qwen-harness chat "Explain dependency injection in one paragraph."
```

Configuration is read from environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible API |
| `OLLAMA_MODEL` | `qwen3.5:4b-cpu` | Installed CPU-only Ollama model tag |
| `HARNESS_MAX_STEPS` | `4` | Maximum model requests per run |
| `HARNESS_TIMEOUT_SECONDS` | `120` | HTTP timeout per request |
| `HARNESS_MAX_OUTPUT_TOKENS` | `512` | Maximum generated tokens per response |

Thinking is disabled by default to keep latency practical on CPU-heavy machines.
The included `Modelfile.cpu` also disables GPU offload because the MX230's CUDA
device was observed crashing Ollama's native model runner. Reboot Windows to
recover a GPU reported as lost; keep using the CPU tag for harness stability.

## Test

```powershell
python -m pytest
```
