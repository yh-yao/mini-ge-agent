# mini-ge-agent

`mini-ge-agent` is a small, readable graph-engineering agent for software tasks. It keeps the single-agent loop minimal, then composes those loops into an inspectable graph with explicit dependencies, deterministic gates, bounded retries, and complete trajectories.

The same `mini` command supports two modes:

```text
loop:   task → agent loop ⇄ shell observation → result

graph:  planner → implementer ⇄ test gate → reviewer ⇄ final gate
```

## Why two modes?

Start with one loop. A graph earns its coordination overhead only when the task needs specialization, independent verification, isolation, or explicit control over handoffs and retries.

| | Loop mode | Graph mode |
| --- | --- | --- |
| Unit of execution | One agent context | Multiple scoped agent loops and deterministic gates |
| Control structure | Observe → act → verify → repeat | Versioned nodes, dependency edges, failure feedback, and gates |
| State | One linear message trajectory | Per-node trajectories plus graph results and attempt history |
| Verification | The agent runs and interprets checks | Test commands can veto progress without an LLM decision |
| Best for | Focused tasks that fit one context | Tasks needing planning, independent review, or grounded retries |
| Main risk | Context drift inside one loop | Handoff loss, coordination cost, and cascading failures |

Multi-agent does not automatically mean graph engineering. Several agents chatting become an engineered graph only when their topology and edge semantics materially control execution and remain inspectable.

## Install

```bash
cd /Users/YuhangYao/Desktop/mini-ge-agent
python -m pip install -e .
mini-ge-extra config setup
```

Python 3.10 or newer is required.

## Run

Loop mode is the default:

```bash
mini --mode loop -m openai/gpt-5.4 -t "Fix the failing tests"
```

Graph mode uses the grounded planner/implementer/reviewer graph:

```bash
mini --mode graph -m openai/gpt-5.4 -t "Implement the requested feature"
```

You can override configuration using built-in YAML files, another YAML path, or key-value settings:

```bash
mini --mode loop -c mini.yaml -c agent.cost_limit=5 -t "Refactor the parser"
mini --mode graph -c ge.yaml -c agent.max_retries=1 -t "Add validation"
```

The default trajectory is stored in the platform-specific `mini-ge-agent` configuration directory as `last_mini_run.traj.json`.

## Loop mode

Loop mode uses the minimal agent control flow in [`src/minigeagent/agents/default.py`](src/minigeagent/agents/default.py):

```python
while True:
    agent.step()  # model query → shell action → observation
```

The harness adds cost, step, wall-time, and format-error limits while preserving a linear, replayable message history. Interactive confirmation is provided by the interactive agent used in the default `mini.yaml` configuration.

## Graph mode

Graph mode uses [`src/minigeagent/agents/graph.py`](src/minigeagent/agents/graph.py) and the default [`ge.yaml`](src/minigeagent/config/ge.yaml) topology:

```text
Planner
   ↓ typed result
Implementer ←──────── failed test output
   ↓                         │ max 2 retries
Deterministic Test Gate ─────┘
   ↓ pass
Reviewer ←────────── final gate failure
   ↓                         │ max 1 retry
Deterministic Final Gate ────┘
```

Agent nodes own independent model contexts. Gate nodes execute frozen shell commands and pass only on a zero exit code. A failed gate sends its concrete output back to its retry target. The runtime, rather than an agent prompt, enforces maximum nodes, maximum retries, total cost, dependencies, and termination.

The graph trajectory records:

- the declared nodes and dependency/control edges;
- each node's final status (`completed`, `failed`, or `blocked`);
- every agent and gate attempt;
- gate return codes and output;
- per-agent conversations, API calls, and cost.

## Customize the graph

Graph nodes are declared in YAML. Agent nodes require a role and task template:

```yaml
- id: implementer
  kind: agent
  role: implementer
  depends_on: [planner]
  task_template: |
    Original task: {{task}}
    Dependencies: {{dependencies}}
    Gate feedback: {{feedback}}
```

Gate nodes execute deterministic evidence checks and can define a bounded retry edge:

```yaml
- id: test_gate
  kind: gate
  depends_on: [implementer]
  command: git diff --check && pytest -q
  retry:
    target: implementer
    max_attempts: 2
```

The retry target must be an agent and a direct dependency of the gate. Cyclic dependencies, unknown nodes, duplicate IDs, and retry limits above the graph policy are rejected before execution.

## Project structure

```text
src/minigeagent/
├── agents/          # loop, interactive, multimodal, and graph control flows
├── environments/    # local and isolated command execution
├── models/          # model-provider interfaces
├── config/          # loop, graph, and benchmark configurations
└── run/             # the mini entry point and benchmark runners
```

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
```

Tests for graph execution cover deterministic passes, gate failures, evidence-backed retries, downstream blocking, topology validation, serialization, and CLI assembly.

## References

This project is based on [`SWE-agent/mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent). It preserves the upstream project's minimal agent loop, model/environment polymorphism, trajectory format conventions, copyright, and MIT notices, then adds graph-engineering primitives and a unified two-mode CLI. The original project and its authors deserve primary credit for the underlying implementation.

Conceptual references:

- [Awesome Graph Engineering](https://github.com/ChaoYue0307/awesome-graph-engineering) — taxonomy and field guide for graph-structured agent systems.
- [From Loop Engineering to Graph Engineering?](https://medium.com/intuitionmachine/from-loop-engineering-to-graph-engineering-d3ebeb08511c) — graph-of-loops framing and the need for external anchors.
- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) — guidance to start with simple, composable agent patterns.
- [A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/) — single-agent and multi-agent orchestration patterns.
- [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823) — language-agent systems represented and optimized as computational graphs.
- [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — failure taxonomy spanning system design, inter-agent alignment, verification, and termination.

## License

New mini-ge-agent contributions are licensed under [Apache License 2.0](LICENSE.md). Code derived from mini-swe-agent retains its upstream MIT terms and notices; see [`NOTICE`](NOTICE) and [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
