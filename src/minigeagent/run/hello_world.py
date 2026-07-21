"""This is the simplest possible example of how to use mini-ge-agent with python bindings.
For a more complete example, see mini.py
"""

import logging
import os
from pathlib import Path

import typer
import yaml

from minigeagent import package_dir
from minigeagent.agents.default import DefaultAgent
from minigeagent.environments.local import LocalEnvironment
from minigeagent.models.litellm_model import LitellmModel

app = typer.Typer()


@app.command()
def main(
    task: str = typer.Option(..., "-t", "--task", help="Task/problem statement", show_default=False, prompt=True),
    model_name: str = typer.Option(
        os.getenv("MGEA_MODEL_NAME"),
        "-m",
        "--model",
        help="Model name (defaults to MGEA_MODEL_NAME env var)",
        prompt="What model do you want to use?",
    ),
) -> DefaultAgent:
    logging.basicConfig(level=logging.DEBUG)
    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        LocalEnvironment(),
        **yaml.safe_load(Path(package_dir / "config" / "default.yaml").read_text())["agent"],
    )
    agent.run(task)
    return agent


if __name__ == "__main__":
    app()
