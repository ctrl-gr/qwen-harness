# Qwen Harness

A small CLI-first agent harness for running `qwen3.5:0.8b` through a local Ollama server.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Make sure Ollama is running and the model is installed:

```powershell
ollama pull qwen3.5:0.8b
ollama create qwen3.5:0.8b-cpu -f .\Modelfile.cpu
```

## Run

```powershell
qwen-harness chat "Explain dependency injection in one paragraph."
```

Add `--verbose` (or `-v`) to follow the observable execution on stderr:

```powershell
qwen-harness chat --verbose "Explain dependency injection in one paragraph."
```

For tasks that must use a workspace tool, declare the completion contract:

```powershell
qwen-harness chat --verbose --require-tool list_files "List the workspace files."
```

`--require-tool` may be repeated. Python marks the run successful only when
every declared tool—and every tool the model actually attempted—has at least
one successful call. A correctable failed call may be retried; a later success
for the same tool satisfies the contract. Otherwise the command exits nonzero
as `INCOMPLETE` and does not print the model's plausible-but-unverified answer.

Verbose events include a run ID, UTC timestamps, state transitions, model and
verification boundaries, errors, and elapsed time. They intentionally exclude
prompts, generated content, credentials, and private model chain-of-thought.
The final response remains on stdout, so it can still be redirected or piped.
Event sinks must return promptly; they are best-effort diagnostics, and sink
exceptions are isolated so logging cannot change a run's outcome.

## Per-run context

Python builds a concise, versioned context for every run. It contains stable
safety invariants and only the completion requirements relevant to the current
task; it does not copy the user's prompt, enumerate irrelevant tools, or retain
conversation history. The `context.built` event records only the context
version, allowing context-template changes to be traced without logging
sensitive text.

This is intentionally small for the 0.8B model: tool schemas describe available
operations, while per-run instructions say when evidence is mandatory. Working
memory and retrieved knowledge will be added later as separate bounded layers
rather than by growing one permanent system prompt.

## Read-only workspace tools

The agent can list files, read bounded UTF-8 text files, and search text under
the directory where `qwen-harness` is launched. Tool paths must be relative;
absolute paths, parent traversal, and symlink or junction escapes are rejected.
There are currently no tools that modify or delete files.

This phase assumes the local workspace is not being maliciously rewritten while
a tool call is in progress. Handle-level protection against concurrent path
replacement belongs to the later isolated-sandbox phase; the current harness
has no mutation tool that can create that race itself.

Future mutation tools follow a backup-first policy: generate and review the
proposed change, create a recoverable backup or reversible patch, verify that
recovery artifact, and only then apply an approved modification.

Configuration is read from environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible API |
| `OLLAMA_MODEL` | `qwen3.5:0.8b-cpu` | Installed CPU-only Ollama model tag |
| `HARNESS_MAX_STEPS` | `4` | Maximum model requests per run |
| `HARNESS_TIMEOUT_SECONDS` | `120` | HTTP timeout per request |
| `HARNESS_MAX_OUTPUT_TOKENS` | `512` | Maximum generated tokens per response |
| `HARNESS_MAX_TOOL_CALLS` | `4` | Maximum tool calls per run |
| `HARNESS_MAX_LIST_ENTRIES` | `200` | Maximum entries returned by one listing |
| `HARNESS_MAX_FILE_BYTES` | `4000` | Maximum bytes read from one text file |
| `HARNESS_MAX_SEARCH_RESULTS` | `8` | Maximum matches returned by one search |
| `HARNESS_MAX_SEARCH_FILES` | `500` | Maximum files visited by one search |
| `HARNESS_MAX_SEARCH_BYTES` | `1000000` | Aggregate bytes examined by one search |
| `HARNESS_MAX_LINE_CHARACTERS` | `200` | Maximum characters returned per matching line |
| `HARNESS_MAX_SEARCH_DIRECTORIES` | `200` | Maximum directories visited by one search |
| `HARNESS_TOOL_TIMEOUT_SECONDS` | `5` | Wall-clock budget for one filesystem tool |

Thinking is disabled by default to keep latency practical on CPU-heavy machines.
The included `Modelfile.cpu` also disables GPU offload because the MX230's CUDA
device was observed crashing Ollama's native model runner. Reboot Windows to
recover a GPU reported as lost; keep using the CPU tag for harness stability.

## Phase 1: Python-owned orchestration

The model returns one typed terminal `finish` decision. Python owns the run
lifecycle and only permits these transitions:

```text
CREATED -> BUILDING_CONTEXT -> CALLING_MODEL -> VERIFYING -> SUCCEEDED
                                                   |
                                                   +-> INCOMPLETE
              Any nonterminal state may transition to FAILED.
```

`Harness.chat(prompt)` remains the simple string-returning interface used by the
CLI. `Harness.run(prompt)` also exposes the validated model decision, final
state, immutable transition trace, and sanitized tool-call evidence. Callers
can pass a `TaskContract` to make successful tool evidence mandatory.

Phase 1 asks Qwen only for plain response text because there is no action to
choose yet. Python wraps that text in the strict terminal decision and verifies
it before marking the run successful. This avoids making the 0.8B model produce
unnecessary tool calls or JSON envelopes.

## Test

```powershell
python -m pytest
```
