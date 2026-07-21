#!/usr/bin/env python3

"""Run mini-ge-agent in loop or graph mode."""

import os
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from minigeagent import global_config_dir
from minigeagent.agents import get_agent
from minigeagent.agents.utils.prompt_user import _multiline_prompt
from minigeagent.config import builtin_config_dir, get_config_from_spec
from minigeagent.environments import get_environment
from minigeagent.models import get_model
from minigeagent.run.utilities.config import configure_if_first_time
from minigeagent.utils.serialize import UNSET, recursive_merge

DEFAULT_LOOP_CONFIG_FILE = Path(os.getenv("MGEA_LOOP_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_GRAPH_CONFIG_FILE = Path(os.getenv("MGEA_GRAPH_CONFIG_PATH", builtin_config_dir / "ge.yaml"))
DEFAULT_CONFIG_FILE = DEFAULT_LOOP_CONFIG_FILE
DEFAULT_OUTPUT_FILE = global_config_dir / "last_mini_run.traj.json"


_HELP_TEXT = """Run mini-ge-agent in your local environment.

[bold]loop[/bold] runs one minimal observe → act → verify agent loop.
[bold]graph[/bold] runs planner and implementation loops connected to deterministic test gates and bounded retries.
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c mini.yaml <other options>[/bold green].
Without this option, the config is selected from --mode: mini.yaml for loop and ge.yaml for graph.

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c mini.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml agent.mode=yolo[/bold green]
"""

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    mode: str = typer.Option("loop", "--mode", help="Execution mode: loop or graph"),
    model_name: Optional[str] = typer.Option(None, "-m", "--model", help="Model to use",),
    model_class: Optional[str] = typer.Option(None, "--model-class", help="Model class to use (e.g., 'litellm' or 'minigeagent.models.litellm_model.LitellmModel')", rich_help_panel="Advanced"),
    agent_class: Optional[str] = typer.Option(None, "--agent-class", help="Agent class to use (e.g., 'interactive' or 'minigeagent.agents.interactive.InteractiveAgent')", rich_help_panel="Advanced"),
    environment_class: Optional[str] = typer.Option(None, "--environment-class", help="Environment class to use (e.g., 'local' or 'minigeagent.environments.local.LocalEnvironment')", rich_help_panel="Advanced"),
    task: Optional[str] = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: Optional[float] = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: list[str] = typer.Option([], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT),
    output: Optional[Path] = typer.Option(DEFAULT_OUTPUT_FILE, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
) -> Any:
    # fmt: on
    configure_if_first_time()

    if mode not in {"loop", "graph"}:
        raise typer.BadParameter("Mode must be 'loop' or 'graph'", param_hint="--mode")
    if not config_spec:
        config_spec = [str(DEFAULT_LOOP_CONFIG_FILE if mode == "loop" else DEFAULT_GRAPH_CONFIG_FILE)]

    # Build the config from the command line arguments
    console.print(f"Building agent config from specs: [bold green]{config_spec}[/bold green]")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "run": {
            "task": task or UNSET,
        },
        "agent": {
            "agent_class": agent_class or UNSET,
            "mode": "yolo" if yolo else UNSET,
            "cost_limit": cost_limit if cost_limit is not None else UNSET,
            "confirm_exit": False if exit_immediately else UNSET,
            "output_path": output or UNSET,
        },
        "model": {
            "model_class": model_class or UNSET,
            "model_name": model_name or UNSET,
        },
        "environment": {
            "environment_class": environment_class or UNSET,
        },
    })
    config = recursive_merge(*configs)

    if (run_task := config.get("run", {}).get("task", UNSET)) is UNSET:
        console.print("[bold yellow]What do you want to do?")
        run_task = _multiline_prompt()
        console.print("[bold green]Got that, thanks![/bold green]")

    model = get_model(config=config.get("model", {}))
    env = get_environment(config.get("environment", {}), default_type="local")
    agent = get_agent(model, env, config.get("agent", {}), default_type="interactive" if mode == "loop" else "graph")
    result = agent.run(run_task)
    if (output_path := config.get("agent", {}).get("output_path")):
        console.print(f"Saved trajectory to [bold green]'{output_path}'[/bold green]")
    if mode == "graph" and result["exit_status"] != "Submitted":
        raise typer.Exit(1)
    return agent


if __name__ == "__main__":
    app()
