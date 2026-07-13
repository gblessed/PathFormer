import os
from typing import List, Sequence

import torch

from models import PathDecoder


DEFAULT_HIDDEN_DIM = 512
DEFAULT_N_LAYERS = 8
DEFAULT_N_HEADS = 8


def generate_paths_no_env_batch(model, prompts, max_steps=25):
    model.eval()
    device = next(model.parameters()).device
    prompts = prompts.to(device)
    batch_size = prompts.size(0)
    cur = torch.zeros(batch_size, 1, 7, device=device, dtype=prompts.dtype)
    inter_str = -1 * torch.ones(batch_size, 1, 4, device=device, dtype=prompts.dtype)
    outputs = []
    outputs_interactions = []

    for _ in range(max_steps):
        (
            delay,
            power,
            _phase_sin,
            _phase_cos,
            phase,
            _az_sin,
            _az_cos,
            az,
            _el_sin,
            _el_cos,
            el,
            _aod_az_sin,
            _aod_az_cos,
            aod_az,
            _aod_el_sin,
            _aod_el_cos,
            aod_el,
            pathcounts,
            interaction_logits,
        ) = model(prompts, cur, inter_str)

        delay_t = delay[:, -1]
        power_t = power[:, -1]
        phase_t = phase[:, -1]
        az_t = az[:, -1]
        el_t = el[:, -1]
        aod_az_t = aod_az[:, -1]
        aod_el_t = aod_el[:, -1]
        interaction_t = (torch.sigmoid(interaction_logits[:, -1]) > 0.5).float()

        next_path = torch.stack(
            [delay_t, power_t, phase_t, az_t, el_t, aod_az_t, aod_el_t],
            dim=-1,
        )
        outputs.append(next_path)
        outputs_interactions.append(interaction_t)
        cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
        inter_str = torch.cat([inter_str, interaction_t.unsqueeze(1)], dim=1)

    return (
        torch.stack(outputs, dim=1).detach().cpu(),
        pathcounts.detach().cpu(),
        torch.stack(outputs_interactions, dim=1).detach().cpu(),
    )


class DirectModelInferenceBackend:
    def __init__(
        self,
        scenario_name: str,
        checkpoint_path: str | None = None,
        checkpoint_dir: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.scenario_name = scenario_name
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = None

    def _default_checkpoint_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "base_no_env")

    def _resolve_checkpoint_path(self) -> str:
        if self.checkpoint_path is not None:
            path = self.checkpoint_path
        else:
            checkpoint_dir = self.checkpoint_dir or self._default_checkpoint_dir()
            path = os.path.join(
                checkpoint_dir,
                f"multiscenario_direct_{self.scenario_name}_best_model_checkpoint.pth",
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    def _load_model(self):
        if self._model is not None:
            return self._model

        model = PathDecoder(
            hidden_dim=DEFAULT_HIDDEN_DIM,
            n_layers=DEFAULT_N_LAYERS,
            n_heads=DEFAULT_N_HEADS,
            include_aod=True,
        ).to(self.device)
        checkpoint = torch.load(self._resolve_checkpoint_path(), map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model
        return model

    def _normalize_prompts(self, prompts: Sequence[Sequence[float]]) -> torch.Tensor:
        if len(prompts) == 0:
            return torch.empty((0, 6), dtype=torch.float32)
        for prompt in prompts:
            if len(prompt) != 6:
                raise ValueError("Each prompt must contain exactly 6 values: [tx_x, tx_y, tx_z, rx_x, rx_y, rx_z].")
        return torch.tensor(prompts, dtype=torch.float32)

    def generate(
        self,
        prompts: Sequence[Sequence[float]],
        max_generate_steps: int = 25,
        return_tensors: bool = False,
    ):
        prompt_tensor = self._normalize_prompts(prompts)
        # if prompt_tensor.numel() == 0:
        #     return {"paths": prompt_tensor, "path_counts": torch.empty((0, 1)), "interactions": torch.empty((0, 0, 4))} if return_tensors else []

        model = self._load_model()
        generated_paths, path_counts, generated_interactions = generate_paths_no_env_batch(
            model,
            prompt_tensor,
            max_steps=max_generate_steps,
        )


        return {
            "paths": generated_paths,
            "path_counts": path_counts,
            "interactions": generated_interactions,
        }
