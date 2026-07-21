import json

import pytest

from minigeagent.agents.graph import GraphAgent
from minigeagent.environments.local import LocalEnvironment
from minigeagent.models.test_models import DeterministicModel, make_output


def node_agent_config() -> dict:
    return {"system_template": "Role node", "instance_template": "{{task}}", "cost_limit": 0}


def submitted(label: str) -> dict:
    return make_output(
        label,
        [{"command": f"printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n{label} evidence\\n'"}],
    )


def test_graph_runs_grounded_pipeline_and_serializes_control_edges(tmp_path):
    output_path = tmp_path / "graph.json"
    agent = GraphAgent(
        DeterministicModel(outputs=[submitted("plan"), submitted("code"), submitted("review")]),
        LocalEnvironment(cwd=str(tmp_path)),
        node_agent=node_agent_config(),
        result_node="review",
        output_path=output_path,
        nodes=[
            {"id": "plan", "role": "planner", "task_template": "Plan {{task}}"},
            {
                "id": "code",
                "role": "implementer",
                "depends_on": ["plan"],
                "task_template": "Implement {{task}} from {{dependencies}} with {{feedback}}",
            },
            {"id": "tests", "kind": "gate", "depends_on": ["code"], "command": "true"},
            {
                "id": "review",
                "role": "reviewer",
                "depends_on": ["tests"],
                "task_template": "Review {{task}} from {{dependencies}} with {{feedback}}",
            },
            {"id": "final", "kind": "gate", "depends_on": ["review"], "command": "true"},
        ],
    )

    assert agent.run("feature") == {
        "exit_status": "Submitted",
        "submission": "review evidence\n",
        "failed_nodes": [],
        "blocked_nodes": [],
    }
    assert "plan evidence" in agent.node_agents["code"].messages[1]["content"]
    assert all(status == "completed" for status in agent.node_status.values())
    assert agent.n_calls == 3
    artifact = json.loads(output_path.read_text())
    assert artifact["trajectory_format"] == "mini-ge-agent-2.0"
    assert artifact["graph"]["attempts"]["tests"][0]["exit_status"] == "Passed"
    assert set(artifact["node_trajectories"]) == {"plan", "code", "review"}
    assert all(len(trajectories) == 1 for trajectories in artifact["node_trajectories"].values())


def test_failed_gate_retries_agent_with_evidence(tmp_path):
    marker = tmp_path / "gate-passed"
    agent = GraphAgent(
        DeterministicModel(outputs=[submitted("first"), submitted("fixed")]),
        LocalEnvironment(cwd=str(tmp_path)),
        node_agent=node_agent_config(),
        nodes=[
            {"id": "worker", "role": "worker", "task_template": "{{task}} {{feedback}}"},
            {
                "id": "gate",
                "kind": "gate",
                "depends_on": ["worker"],
                "command": f"test -f '{marker}' || (touch '{marker}' && false)",
                "retry": {"target": "worker", "max_attempts": 1},
            },
        ],
    )

    assert agent.run("task")["exit_status"] == "Submitted"
    assert len(agent.node_attempts["worker"]) == 2
    assert [attempt["exit_status"] for attempt in agent.node_attempts["gate"]] == ["Failed", "Passed"]
    assert '"exit_status": "Failed"' in agent.node_agents["worker"].messages[1]["content"]
    assert agent.n_calls == 2
    assert agent.serialize()["graph"]["edges"][-1] == {
        "source": "gate",
        "target": "worker",
        "payload": "gate_failure",
        "control": "retry",
        "max_attempts": 1,
    }


def test_gate_failure_blocks_dependents():
    agent = GraphAgent(
        DeterministicModel(outputs=[submitted("work")]),
        LocalEnvironment(),
        node_agent=node_agent_config(),
        nodes=[
            {"id": "worker", "role": "worker", "task_template": "{{task}}"},
            {"id": "gate", "kind": "gate", "depends_on": ["worker"], "command": "false"},
            {"id": "review", "role": "reviewer", "depends_on": ["gate"], "task_template": "{{task}}"},
        ],
    )

    assert agent.run("task") == {
        "exit_status": "GraphFailed",
        "submission": "",
        "failed_nodes": ["gate"],
        "blocked_nodes": ["review"],
    }
    assert agent.node_status == {"worker": "completed", "gate": "failed", "review": "blocked"}
    assert agent.n_calls == 1


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        ([{"id": "same", "role": "a", "task_template": "x"}, {"id": "same", "role": "b", "task_template": "x"}], "unique"),
        ([{"id": "a", "role": "a", "task_template": "x", "depends_on": ["missing"]}], "Unknown"),
        (
            [
                {"id": "a", "role": "a", "task_template": "x", "depends_on": ["b"]},
                {"id": "b", "role": "b", "task_template": "x", "depends_on": ["a"]},
            ],
            "cycle",
        ),
        (
            [
                {"id": "worker", "role": "worker", "task_template": "x"},
                {
                    "id": "gate",
                    "kind": "gate",
                    "depends_on": ["worker"],
                    "command": "true",
                    "retry": {"target": "missing"},
                },
            ],
            "Retry target",
        ),
    ],
)
def test_graph_rejects_invalid_topologies(nodes, message):
    with pytest.raises(ValueError, match=message):
        GraphAgent(DeterministicModel(outputs=[]), LocalEnvironment(), node_agent=node_agent_config(), nodes=nodes)
