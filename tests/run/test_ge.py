import json

from minigeagent.run.mini import main


def test_ge_run_with_deterministic_model(tmp_path, monkeypatch):
    config_path = tmp_path / "ge.yaml"
    output_path = tmp_path / "trajectory.json"
    config_path.write_text(
        """
agent:
  agent_class: graph
  node_agent:
    system_template: node
    instance_template: '{{task}}'
    cost_limit: 0
  nodes:
    - id: worker
      role: worker
      task_template: 'Solve {{task}}'
environment:
  cwd: PLACEHOLDER
model:
  model_class: deterministic
  model_name: deterministic
  outputs:
    - role: assistant
      content: done
      extra:
        actions:
          - command: "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nverified\\n'"
        cost: 1.0
""".replace("PLACEHOLDER", str(tmp_path))
    )
    monkeypatch.setenv("MGEA_CONFIGURED", "true")

    agent = main(
        mode="graph",
        model_name=None,
        task="a feature",
        yolo=False,
        cost_limit=None,
        config_spec=[str(config_path)],
        output=output_path,
        model_class=None,
        agent_class=None,
        environment_class=None,
        exit_immediately=False,
    )

    assert agent.result["submission"] == "verified\n"
    assert json.loads(output_path.read_text())["graph"]["nodes"][0]["status"] == "completed"
    assert json.loads(output_path.read_text())["trajectory_format"] == "mini-ge-agent-2.0"
