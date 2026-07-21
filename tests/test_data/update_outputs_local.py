#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from unittest.mock import patch

from minigeagent.models.test_models import DeterministicModel
from minigeagent.run.mini import DEFAULT_CONFIG_FILE, main


def update_trajectory():
    traj_path = Path(__file__).parent / "local.traj.json"
    trajectory = json.loads(traj_path.read_text())

    task = "Blah blah blah"

    model_responses = [msg["content"] for msg in trajectory[2:] if msg["role"] == "assistant"]
    print(f"Got {len(model_responses)} model responses")

    with patch("minigeagent.run.mini.get_model") as mock_get_model:
        mock_get_model.return_value = DeterministicModel(outputs=model_responses)
        main(
            mode="loop",
            model_name="tardis",
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            output=traj_path,
            task=task,
            yolo=True,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

if __name__ == "__main__":
    update_trajectory()
