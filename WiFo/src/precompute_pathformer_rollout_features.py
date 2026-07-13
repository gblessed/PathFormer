import argparse
import os
import sys
import warnings

import deepmimo as dm
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATHFORMER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if PATHFORMER_ROOT not in sys.path:
    sys.path.insert(0, PATHFORMER_ROOT)

from dataset.dataloaders import PreTrainMySeqDataLoader
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    generate_paths_first_step_residual_batch,
    load_best_checkpoint as load_residual_checkpoint,
)
from multiscenario_direct_training_first_step_residual_corridor import (
    build_first_step_assignments_with_corridor,
)
from scene_feature_utils import SceneFeatureBank
from utils.utils import ChannelParameters, MyChannelComputer

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_CHECKPOINT = (
    "/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor_concat/"
    "first_step_residual_corridor_concat_27scenarios_44710a4a_best_model_checkpoint.pth"
)
DEFAULT_DATA_DIR = "/home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute frozen PathFormer rollout embeddings aligned to dynamic WiFo datasets."
    )
    parser.add_argument("--scenario", type=str, default=None, help="Run only one scenario.")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Optional explicit scenario list.")
    parser.add_argument("--all-scenarios", action="store_true", help="Process every scenario found in the data dir.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to --data-dir.")
    parser.add_argument("--pathformer-checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-generated-steps", type=int, default=25)
    parser.add_argument("--embed-pool", choices=["first", "last", "mean"], default="mean")
    parser.add_argument("--prior-num-paths", type=int, default=10)
    parser.add_argument("--prior-time-steps", type=int, default=8)
    parser.add_argument("--prior-t-patch-size", type=int, default=4)
    parser.add_argument("--prior-patch-size", type=int, default=4)
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def tx_key_from_array(tx_pos):
    tx_pos = np.asarray(tx_pos, dtype=np.float32).reshape(-1)
    return tuple(np.round(tx_pos, 6).tolist())


def resolve_scenarios(args):
    if args.scenarios:
        return args.scenarios
    if args.scenario:
        return [args.scenario]
    if args.all_scenarios:
        scenario_names = []
        for filename in os.listdir(args.data_dir):
            if filename.startswith("_") and filename.endswith("_train_data.pt"):
                scenario_names.append(filename[len("_") : -len("_train_data.pt")])
        return sorted(set(scenario_names))
    raise ValueError("Select scenarios with --scenario, --scenarios, or --all-scenarios.")


def maybe_listify_dataset(dataset):
    if hasattr(dataset, "n_ue") and isinstance(dataset.n_ue, int):
        return [dataset]
    return list(dataset)


def build_split_user_index_mapping(dataset, train, seed, train_ratio):
    dataset_list = maybe_listify_dataset(dataset)
    mapping = []
    for tx_idx, data_tx in enumerate(dataset_list):
        indices = np.arange(data_tx.n_ue)
        np.random.seed(seed + tx_idx)
        np.random.shuffle(indices)
        split_idx = int(train_ratio * len(indices))
        if train:
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]
        use_indices = data_tx.los != -1
        indices = [int(i) for i in indices if use_indices[i]]
        tx_pos_arr = np.asarray(data_tx.tx_pos, dtype=np.float32)
        tx_pos = tx_pos_arr if tx_pos_arr.ndim == 1 else tx_pos_arr[0]
        tx_key = tx_key_from_array(tx_pos)
        mapping.extend((tx_key, user_idx) for user_idx in indices)
    return mapping


class PathFormerRolloutEmbedder(torch.nn.Module):
    def __init__(self, path_model, pool_mode="mean", max_generated_steps=25):
        super().__init__()
        self.path_model = path_model
        self.pool_mode = pool_mode
        self.max_generated_steps = max_generated_steps
        for param in self.path_model.parameters():
            param.requires_grad = False

    def _forward_hidden(self, prompts, paths, interactions):
        return self.path_model.backbone.forward_hidden(prompts, paths, interactions)[0]

    def _predict_next(self, prompts, paths, interactions, first_step_baseline):
        outputs = self.path_model(prompts, paths, interactions, first_step_baseline)
        delay = outputs[0][:, -1]
        power = outputs[1][:, -1]
        phase = outputs[4][:, -1]
        aoa_az = outputs[7][:, -1]
        aoa_el = outputs[10][:, -1]
        aod_az = outputs[13][:, -1]
        aod_el = outputs[16][:, -1]
        interaction = torch.sigmoid(outputs[-1][:, -1, :])
        next_path = torch.stack([delay, power, phase, aoa_az, aoa_el, aod_az, aod_el], dim=-1)
        return next_path, interaction

    def _rollout_hidden(self, prompts, steps, first_step_baseline):
        batch_size = prompts.size(0)
        cur = torch.zeros(batch_size, 1, 7, device=prompts.device, dtype=prompts.dtype)
        interactions = -1 * torch.ones(batch_size, 1, 4, device=prompts.device, dtype=prompts.dtype)
        hidden_steps = []

        for _ in range(steps):
            h_paths = self._forward_hidden(prompts, cur, interactions)
            hidden_steps.append(h_paths[:, -1, :])
            next_path, next_interaction = self._predict_next(
                prompts,
                cur,
                interactions,
                first_step_baseline,
            )
            cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
            interactions = torch.cat([interactions, next_interaction.unsqueeze(1)], dim=1)

        return torch.stack(hidden_steps, dim=1)

    def rollout_hidden_and_paths(self, prompts, first_step_baseline):
        batch_size = prompts.size(0)
        cur = torch.zeros(batch_size, 1, 7, device=prompts.device, dtype=prompts.dtype)
        interactions = -1 * torch.ones(batch_size, 1, 4, device=prompts.device, dtype=prompts.dtype)
        hidden_steps = []
        generated_steps = []

        for _ in range(self.max_generated_steps):
            h_paths = self._forward_hidden(prompts, cur, interactions)
            hidden_steps.append(h_paths[:, -1, :])
            next_path, next_interaction = self._predict_next(
                prompts,
                cur,
                interactions,
                first_step_baseline,
            )
            generated_steps.append(next_path)
            cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
            interactions = torch.cat([interactions, next_interaction.unsqueeze(1)], dim=1)

        return torch.stack(hidden_steps, dim=1), torch.stack(generated_steps, dim=1)

    def _pool(self, h_paths):
        if self.pool_mode == "first":
            return h_paths[:, 0, :]
        if self.pool_mode == "last":
            return h_paths[:, -1, :]
        return h_paths.mean(dim=1)

    @torch.no_grad()
    def extract_features(self, prompts, first_step_baseline):
        h_paths = self._rollout_hidden(
            prompts,
            self.max_generated_steps,
            first_step_baseline,
        )
        return self._pool(h_paths)

    @torch.no_grad()
    def extract_token_features(self, prompts, first_step_baseline):
        return self._rollout_hidden(
            prompts,
            self.max_generated_steps,
            first_step_baseline,
        )


def build_pathformer_datasets(dataset, args):
    pad_value = 0
    base_train = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    base_val = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    scene_bank = SceneFeatureBank.from_dataset(dataset, use_material_features=args.use_material_features)
    train_aug_prompts, train_baselines, val_aug_prompts, val_baselines = build_first_step_assignments_with_corridor(
        base_train,
        base_val,
        scene_bank,
        n_clusters=args.n_clusters,
        nearest_k=args.nearest_k,
        corridor_k=args.corridor_k,
        corridor_bins=args.corridor_bins,
    )
    train_data = FirstStepResidualDataset(base_train, train_aug_prompts, train_baselines)
    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
    return train_data, val_data


@torch.no_grad()
def precompute_features(embedder, dataset, batch_size, device, desc):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    all_features = []
    all_token_features = []
    for batch in tqdm(loader, desc=desc, leave=False):
        prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines = batch
        prompts = prompts.to(device)
        first_step_baselines = first_step_baselines.to(device)
        features = embedder.extract_features(prompts, first_step_baselines)
        token_features = embedder.extract_token_features(prompts, first_step_baselines)
        all_features.append(features.cpu())
        all_token_features.append(token_features.cpu())
    return torch.cat(all_features, dim=0), torch.cat(all_token_features, dim=0)


def build_feature_lookup(feature_tensor, ordered_keys):
    if feature_tensor.shape[0] != len(ordered_keys):
        raise ValueError(
            f"Feature tensor rows ({feature_tensor.shape[0]}) do not match mapping entries ({len(ordered_keys)})."
        )
    return {
        key: feature_tensor[idx]
        for idx, key in enumerate(ordered_keys)
    }


def _power_scaled_to_linear(power_scaled):
    return 10 ** ((np.clip(power_scaled, -15000.0, 500.0) / 0.01) / 10)


def patchify_magnitude_prior(prior_mag, t_patch_size, patch_size):
    """Patchify [B, T, H, W] magnitude priors into WiFo-aligned real tokens."""
    if prior_mag.dim() != 4:
        raise ValueError(f"Expected prior_mag [B,T,H,W], got {tuple(prior_mag.shape)}")
    batch, time_steps, height, width = prior_mag.shape
    if time_steps % t_patch_size != 0 or height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Prior shape {(time_steps, height, width)} is not divisible by "
            f"t_patch_size={t_patch_size}, patch_size={patch_size}."
        )
    t = time_steps // t_patch_size
    h = height // patch_size
    w = width // patch_size
    x = prior_mag.reshape(batch, t, t_patch_size, h, patch_size, w, patch_size)
    x = torch.einsum("ntuhpwq->nthwupq", x)
    return x.reshape(batch, t * h * w, t_patch_size * patch_size * patch_size)


def generated_paths_to_prior_tokens(generated, args):
    pred_np = generated[:, : args.prior_num_paths, :].detach().cpu().numpy().astype(np.float32)
    computer = MyChannelComputer()
    params = ChannelParameters()
    batch_size_actual = pred_np.shape[0]
    n_paths = pred_np.shape[1]

    power = _power_scaled_to_linear(pred_np[:, :, 1])
    delay = pred_np[:, :, 0] / 1e6
    phase = np.zeros((batch_size_actual, n_paths), dtype=np.float32)
    aod_az = pred_np[:, :, 5]
    aod_el = pred_np[:, :, 6]
    channel = computer.compute_channels(power, delay, phase, aod_az, aod_el, params=params)
    channel_mag = torch.from_numpy(np.abs(channel).astype(np.float32))

    if channel_mag.dim() != 4:
        raise ValueError(f"Expected synthesized channel [B,RX,TX,SC], got {tuple(channel_mag.shape)}")

    prior_mag = channel_mag[:, 0, :, :].unsqueeze(1).repeat(1, int(args.prior_time_steps), 1, 1)
    prior_mag = torch.log1p(prior_mag)
    flat = prior_mag.flatten(start_dim=1)
    mean = flat.mean(dim=1).view(-1, 1, 1, 1)
    std = flat.std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    prior_mag = (prior_mag - mean) / std
    return patchify_magnitude_prior(
        prior_mag,
        t_patch_size=args.prior_t_patch_size,
        patch_size=args.prior_patch_size,
    )


@torch.no_grad()
def precompute_all_features(embedder, dataset, batch_size, device, args, desc):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    feature_batches = []
    token_batches = []
    prior_batches = []

    for batch in tqdm(loader, desc=desc, leave=False):
        prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines = batch
        del paths, path_lengths, interactions, env, env_prop, path_padding_mask
        prompts = prompts.to(device)
        first_step_baselines = first_step_baselines.to(device)
        h_paths, generated = embedder.rollout_hidden_and_paths(
            prompts,
            first_step_baselines,
        )
        feature_batches.append(embedder._pool(h_paths).cpu())
        token_batches.append(h_paths.cpu())
        prior_batches.append(generated_paths_to_prior_tokens(generated, args).cpu())

    return (
        torch.cat(feature_batches, dim=0).to(torch.float32),
        torch.cat(token_batches, dim=0).to(torch.float32),
        torch.cat(prior_batches, dim=0).to(torch.float32),
    )


def load_dynamic_payload(args, scenario, split):
    path = os.path.join(args.data_dir, f"_{scenario}_{split}_data.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dynamic dataset: {path}")
    return path, torch.load(path, map_location="cpu", weights_only=False)


def build_sidecar_payload(dynamic_payload, pooled_lookup, token_lookup, prior_token_lookup=None):
    tx_positions = dynamic_payload["tx_positions"].cpu().numpy()
    user_indices = dynamic_payload["trajectory_user_indices"].cpu().numpy()
    pooled_rows = []
    token_rows = []
    prior_token_rows = []
    for sample_idx in range(tx_positions.shape[0]):
        tx_key = tx_key_from_array(tx_positions[sample_idx])
        user_idx = int(user_indices[sample_idx, -1])
        lookup_key = (tx_key, user_idx)
        if lookup_key not in pooled_lookup or lookup_key not in token_lookup:
            raise KeyError(
                f"Could not find PathFormer feature for tx={tx_key}, user_idx={user_idx}."
            )
        pooled_rows.append(pooled_lookup[lookup_key].unsqueeze(0))
        token_rows.append(token_lookup[lookup_key].unsqueeze(0))
        if prior_token_lookup is not None:
            if lookup_key not in prior_token_lookup:
                raise KeyError(
                    f"Could not find PathFormer prior token for tx={tx_key}, user_idx={user_idx}."
                )
            prior_token_rows.append(prior_token_lookup[lookup_key].unsqueeze(0))
    prior_tokens = None
    if prior_token_rows:
        prior_tokens = torch.cat(prior_token_rows, dim=0).to(torch.float32)
    return (
        torch.cat(pooled_rows, dim=0).to(torch.float32),
        torch.cat(token_rows, dim=0).to(torch.float32),
        prior_tokens,
    )


def save_feature_sidecar(path, scenario, split, pooled_features, token_features, prior_token_features, args):
    payload = {
        "scenario": scenario,
        "split": split,
        "pathformer_features": pooled_features,
        "pathformer_token_features": token_features,
        "pathformer_checkpoint": args.pathformer_checkpoint,
        "embed_pool": args.embed_pool,
        "max_generated_steps": args.max_generated_steps,
    }
    if prior_token_features is not None:
        payload["pathformer_prior_token_features"] = prior_token_features
        payload["pathformer_prior_kind"] = "log_abs_zero_phase_channel"
        payload["pathformer_prior_num_paths"] = args.prior_num_paths
        payload["pathformer_prior_patch_size"] = args.prior_patch_size
        payload["pathformer_prior_t_patch_size"] = args.prior_t_patch_size
    torch.save(payload, path)


def run_scenario(args, scenario, device):
    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    train_sidecar = os.path.join(output_dir, f"_{scenario}_train_pathformer_features.pt")
    val_sidecar = os.path.join(output_dir, f"_{scenario}_val_pathformer_features.pt")
    if not args.overwrite and os.path.exists(train_sidecar) and os.path.exists(val_sidecar):
        print(f"[{scenario}] feature sidecars already exist, skipping.")
        return

    print(f"\n[{scenario}] loading scenario and building PathFormer rollout features")
    dataset = dm.load(scenario)
    train_data, val_data = build_pathformer_datasets(dataset, args)

    prompt_dim = int(train_data.augmented_prompts[0].numel())
    pathformer = FirstStepResidualPathDecoder(
        prompt_dim=prompt_dim,
        hidden_dim=1024,
        n_layers=12,
        n_heads=8,
    ).to(device)
    load_residual_checkpoint(pathformer, args.pathformer_checkpoint)
    embedder = PathFormerRolloutEmbedder(
        path_model=pathformer,
        pool_mode=args.embed_pool,
        max_generated_steps=args.max_generated_steps,
    ).to(device)
    embedder.eval()

    train_features, train_token_features, train_prior_token_features = precompute_all_features(
        embedder,
        train_data,
        args.batch_size,
        device,
        args,
        desc=f"PathFormer train features/prior [{scenario}]",
    )
    val_features, val_token_features, val_prior_token_features = precompute_all_features(
        embedder,
        val_data,
        args.batch_size,
        device,
        args,
        desc=f"PathFormer val features/prior [{scenario}]",
    )

    train_mapping = build_split_user_index_mapping(dataset, train=True, seed=args.seed, train_ratio=args.train_ratio)
    val_mapping = build_split_user_index_mapping(dataset, train=False, seed=args.seed, train_ratio=args.train_ratio)

    train_lookup = build_feature_lookup(
        train_features,
        train_mapping,
    )
    train_token_lookup = build_feature_lookup(
        train_token_features,
        train_mapping,
    )
    train_prior_token_lookup = build_feature_lookup(
        train_prior_token_features,
        train_mapping,
    )
    val_lookup = build_feature_lookup(
        val_features,
        val_mapping,
    )
    val_token_lookup = build_feature_lookup(
        val_token_features,
        val_mapping,
    )
    val_prior_token_lookup = build_feature_lookup(
        val_prior_token_features,
        val_mapping,
    )

    _, dynamic_train = load_dynamic_payload(args, scenario, "train")
    _, dynamic_val = load_dynamic_payload(args, scenario, "val")
    train_sidecar_features, train_sidecar_token_features, train_sidecar_prior_token_features = build_sidecar_payload(
        dynamic_train,
        train_lookup,
        train_token_lookup,
        train_prior_token_lookup,
    )
    val_sidecar_features, val_sidecar_token_features, val_sidecar_prior_token_features = build_sidecar_payload(
        dynamic_val,
        val_lookup,
        val_token_lookup,
        val_prior_token_lookup,
    )

    save_feature_sidecar(
        train_sidecar,
        scenario,
        "train",
        train_sidecar_features,
        train_sidecar_token_features,
        train_sidecar_prior_token_features,
        args,
    )
    save_feature_sidecar(
        val_sidecar,
        scenario,
        "val",
        val_sidecar_features,
        val_sidecar_token_features,
        val_sidecar_prior_token_features,
        args,
    )
    print(
        f"[{scenario}] saved pooled train {tuple(train_sidecar_features.shape)} "
        f"token train {tuple(train_sidecar_token_features.shape)} "
        f"prior token train {tuple(train_sidecar_prior_token_features.shape)} "
        f"and pooled val {tuple(val_sidecar_features.shape)} "
        f"token val {tuple(val_sidecar_token_features.shape)} "
        f"prior token val {tuple(val_sidecar_prior_token_features.shape)}"
    )


def main():
    args = parse_args()
    if not os.path.exists(args.pathformer_checkpoint):
        raise FileNotFoundError(f"PathFormer checkpoint not found: {args.pathformer_checkpoint}")
    scenarios = resolve_scenarios(args)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Precomputing PathFormer rollout features for {len(scenarios)} scenario(s): {scenarios}")
    for scenario in scenarios:
        run_scenario(args, scenario, device)


if __name__ == "__main__":
    main()
