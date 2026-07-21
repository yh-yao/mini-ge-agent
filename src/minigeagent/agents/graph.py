"""A small graph of agent loops grounded by deterministic gates."""

import json
import logging
from pathlib import Path
from typing import Literal

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field, model_validator

from minigeagent import Environment, Model, __version__
from minigeagent.agents.default import DefaultAgent
from minigeagent.utils.serialize import recursive_merge


class RetryConfig(BaseModel):
    target: str
    max_attempts: int = Field(default=1, ge=0)


class GraphNodeConfig(BaseModel):
    id: str
    kind: Literal["agent", "gate"] = "agent"
    role: str = ""
    task_template: str = ""
    command: str = ""
    depends_on: list[str] = Field(default_factory=list)
    retry: RetryConfig | None = None
    agent: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_kind(self):
        if self.kind == "agent" and (not self.role or not self.task_template or self.command or self.retry):
            raise ValueError("Agent nodes require role and task_template, and cannot define command or retry")
        if self.kind == "gate" and (not self.command or self.role or self.task_template or self.agent):
            raise ValueError("Gate nodes require command, and cannot define agent fields")
        return self


class GraphAgentConfig(BaseModel):
    nodes: list[GraphNodeConfig]
    node_agent: dict
    result_node: str | None = None
    max_nodes: int = Field(default=8, ge=1)
    max_retries: int = Field(default=2, ge=0)
    cost_limit: float = 0.0
    output_path: Path | None = None

    @model_validator(mode="after")
    def validate_graph(self):
        ids = [node.id for node in self.nodes]
        if not ids:
            raise ValueError("Graph must contain at least one node")
        if len(ids) > self.max_nodes:
            raise ValueError(f"Graph has {len(ids)} nodes, exceeding max_nodes={self.max_nodes}")
        if len(ids) != len(set(ids)):
            raise ValueError("Graph node ids must be unique")
        nodes = {node.id: node for node in self.nodes}
        unknown = {dependency for node in self.nodes for dependency in node.depends_on} - set(ids)
        if unknown:
            raise ValueError(f"Unknown graph dependencies: {', '.join(sorted(unknown))}")
        if self.result_node and self.result_node not in nodes:
            raise ValueError(f"Unknown result_node: {self.result_node}")
        for node in self.nodes:
            if not node.retry:
                continue
            if node.retry.target not in nodes or nodes[node.retry.target].kind != "agent":
                raise ValueError(f"Retry target for {node.id} must be an agent node")
            if node.retry.target not in node.depends_on:
                raise ValueError(f"Retry target for {node.id} must be a direct dependency")
            if node.retry.max_attempts > self.max_retries:
                raise ValueError(f"Retry attempts for {node.id} exceed max_retries={self.max_retries}")
        _topological_order(self.nodes)
        return self


def _topological_order(nodes: list[GraphNodeConfig]) -> list[GraphNodeConfig]:
    ordered: list[GraphNodeConfig] = []
    remaining = list(nodes)
    while remaining:
        ready = [node for node in remaining if set(node.depends_on) <= {item.id for item in ordered}]
        if not ready:
            raise ValueError("Graph contains a dependency cycle")
        ordered.extend(ready)
        remaining = [node for node in remaining if node not in ready]
    return ordered


class GraphAgent:
    """Run agent nodes and let deterministic gates veto or retry them."""

    def __init__(self, model: Model, env: Environment, *, config_class: type = GraphAgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.model = model
        self.env = env
        self.logger = logging.getLogger("graph_agent")
        self.node_agents: dict[str, DefaultAgent] = {}
        self.agent_runs: list[tuple[str, DefaultAgent]] = []
        self.node_results: dict[str, dict] = {}
        self.node_attempts: dict[str, list[dict]] = {node.id: [] for node in self.config.nodes}
        self.node_status = {node.id: "pending" for node in self.config.nodes}
        self.result: dict = {}

    @property
    def cost(self) -> float:
        return sum(agent.cost for _, agent in self.agent_runs)

    @property
    def n_calls(self) -> int:
        return sum(agent.n_calls for _, agent in self.agent_runs)

    def run(self, task: str, **kwargs) -> dict:
        self.node_agents = {}
        self.agent_runs = []
        self.node_results = {}
        self.node_attempts = {node.id: [] for node in self.config.nodes}
        self.node_status = {node.id: "pending" for node in self.config.nodes}
        self.result = {}
        for node in _topological_order(self.config.nodes):
            if any(self.node_status[dependency] != "completed" for dependency in node.depends_on):
                self.node_status[node.id] = "blocked"
            elif node.kind == "agent":
                self._run_agent(node, task, {}, kwargs)
            else:
                self._run_gate(node, task, kwargs)
            self.save(self.config.output_path)
        failed = [node_id for node_id, status in self.node_status.items() if status == "failed"]
        blocked = [node_id for node_id, status in self.node_status.items() if status == "blocked"]
        result_node = self.config.result_node or self.config.nodes[-1].id
        self.result = {
            "exit_status": "GraphFailed" if failed or blocked else "Submitted",
            "submission": self.node_results.get(result_node, {}).get("submission", ""),
            "failed_nodes": failed,
            "blocked_nodes": blocked,
        }
        self.save(self.config.output_path)
        return self.result

    def _run_agent(self, node: GraphNodeConfig, task: str, feedback: dict, kwargs: dict) -> None:
        if 0 < self.config.cost_limit <= self.cost:
            self.node_status[node.id] = "blocked"
            return
        self.node_status[node.id] = "running"
        self.save(self.config.output_path)
        node_task = Template(node.task_template, undefined=StrictUndefined).render(
            task=task,
            role=node.role,
            dependencies=json.dumps(
                {dependency: self.node_results[dependency] for dependency in node.depends_on},
                ensure_ascii=False,
                indent=2,
            ),
            feedback=json.dumps(feedback, ensure_ascii=False, indent=2),
            **kwargs,
        )
        self.logger.info("Running agent node %s (%s)", node.id, node.role)
        agent = DefaultAgent(
            self.model,
            self.env,
            **recursive_merge(self.config.node_agent, node.agent, {"output_path": None}),
        )
        self.node_agents[node.id] = agent
        self.agent_runs.append((node.id, agent))
        self.node_results[node.id] = agent.run(node_task)
        self.node_attempts[node.id].append(self.node_results[node.id])
        self.node_status[node.id] = (
            "completed" if self.node_results[node.id].get("exit_status") == "Submitted" else "failed"
        )

    def _run_gate(self, node: GraphNodeConfig, task: str, kwargs: dict) -> None:
        self._execute_gate(node)
        for _ in range(node.retry.max_attempts if node.retry else 0):
            if self.node_status[node.id] == "completed":
                return
            target = next(candidate for candidate in self.config.nodes if candidate.id == node.retry.target)
            self._run_agent(target, task, {node.id: self.node_results[node.id]}, kwargs)
            if self.node_status[target.id] != "completed":
                self.node_status[node.id] = "blocked"
                return
            self._execute_gate(node)

    def _execute_gate(self, node: GraphNodeConfig) -> None:
        self.node_status[node.id] = "running"
        self.save(self.config.output_path)
        self.logger.info("Running deterministic gate %s", node.id)
        output = self.env.execute({"command": node.command})
        self.node_results[node.id] = {
            "exit_status": "Passed" if output["returncode"] == 0 else "Failed",
            "submission": output.get("output", ""),
            "returncode": output["returncode"],
            "exception_info": output.get("exception_info", ""),
        }
        self.node_attempts[node.id].append(self.node_results[node.id])
        self.node_status[node.id] = "completed" if output["returncode"] == 0 else "failed"

    def serialize(self, *extra_dicts) -> dict:
        edges = [
            {"source": dependency, "target": node.id, "payload": "result", "control": "dependency"}
            for node in self.config.nodes
            for dependency in node.depends_on
        ] + [
            {
                "source": node.id,
                "target": node.retry.target,
                "payload": "gate_failure",
                "control": "retry",
                "max_attempts": node.retry.max_attempts,
            }
            for node in self.config.nodes
            if node.retry
        ]
        data = {
            "info": {
                "model_stats": {"instance_cost": self.cost, "api_calls": self.n_calls},
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                **self.result,
            },
            "graph": {
                "nodes": [
                    {**node.model_dump(mode="json"), "status": self.node_status[node.id]}
                    for node in self.config.nodes
                ],
                "edges": edges,
                "results": self.node_results,
                "attempts": self.node_attempts,
            },
            "node_trajectories": {
                node.id: [agent.serialize() for node_id, agent in self.agent_runs if node_id == node.id]
                for node in self.config.nodes
                if node.kind == "agent"
            },
            "trajectory_format": "mini-ge-agent-2.0",
        }
        return recursive_merge(data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
