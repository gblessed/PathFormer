import argparse
from pathlib import Path

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder
from utils.utils import generate_paths_no_env_batch, masked_loss


DEFAULT_SCENARIOS = [
    "city_47_chicago_3p5",
]

PAD_VALUE = 0.0
MAX_GENERATE_STEPS = 25


class PromptToCodebookMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_clusters):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_clusters),
        )

    def forward(self, x):
        return self.net(x)


def _extract_sample(item):
    prompt = item[0]
    paths = item[1]
    path_length = item[2]
    interactions = item[3]
    if paths.shape[0] <= 1:
        return None

    return {
        "prompt": prompt.numpy().astype(np.float32),
        "paths": paths.numpy().astype(np.float32),
        "interactions": interactions.numpy().astype(np.float32),
        "path_length": path_length.numpy().astype(np.float32),
    }


def _collect_samples(seq_data):
    samples = []
    for item in seq_data:
        sample = _extract_sample(item)
        if sample is not None:
            samples.append(sample)
    return samples


def _pad_sequence_batch(paths_list, interactions_list):
    n_samples = len(paths_list)
    max_len = MAX_GENERATE_STEPS + 1

    padded_paths = np.full((n_samples, max_len, 5), PAD_VALUE, dtype=np.float32)
    padded_interactions = -np.ones((n_samples, max_len, 4), dtype=np.float32)
    path_padding_mask = np.zeros((n_samples, max_len), dtype=bool)

    for idx, (paths, interactions) in enumerate(zip(paths_list, interactions_list)):
        seq_len = min(paths.shape[0], max_len)
        padded_paths[idx, :seq_len] = paths[:seq_len]
        padded_interactions[idx, :seq_len] = interactions[:seq_len]
        path_padding_mask[idx, :seq_len] = True

    return padded_paths, padded_interactions, path_padding_mask


def _fit_standardizer(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1e-6, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def _build_sequence_vectors(paths_list, path_lengths):
    vectors = []
    for paths, path_length in zip(paths_list, path_lengths):
        padded = np.zeros((MAX_GENERATE_STEPS, 5), dtype=np.float32)
        n_valid = int(round(float(np.squeeze(path_length)) * MAX_GENERATE_STEPS))
        n_valid = max(0, min(n_valid, MAX_GENERATE_STEPS))
        if n_valid > 0:
            padded[:n_valid] = paths[1:1 + n_valid, :5]
        vec = np.concatenate(
            [padded.reshape(-1), np.array([float(n_valid)], dtype=np.float32)],
            axis=0,
        )
        vectors.append(vec.astype(np.float32))
    return np.stack(vectors, axis=0).astype(np.float32)


def _prepare_data(seq_data):
    samples = _collect_samples(seq_data)
    prompts = np.stack([s["prompt"] for s in samples], axis=0).astype(np.float32)
    paths_list = [s["paths"] for s in samples]
    interactions_list = [s["interactions"] for s in samples]
    path_lengths = np.stack([s["path_length"] for s in samples], axis=0).astype(np.float32)
    padded_paths, padded_interactions, path_padding_mask = _pad_sequence_batch(paths_list, interactions_list)
    seq_vectors = _build_sequence_vectors(paths_list, path_lengths)
    return {
        "prompts": prompts,
        "paths": padded_paths,
        "interactions": padded_interactions,
        "path_lengths": path_lengths,
        "path_padding_mask": path_padding_mask,
        "seq_vectors": seq_vectors,
    }


def _fit_codebook(train_seq_vectors, latent_dim, n_clusters, random_state):
    seq_mean, seq_std = _fit_standardizer(train_seq_vectors)
    train_seq_vectors_norm = _apply_standardizer(train_seq_vectors, seq_mean, seq_std)

    latent_dim_eff = min(latent_dim, train_seq_vectors_norm.shape[1], train_seq_vectors_norm.shape[0])
    if latent_dim_eff <= 0:
        raise ValueError("Invalid latent dimension.")

    pca = PCA(n_components=latent_dim_eff, random_state=random_state)
    train_latents = pca.fit_transform(train_seq_vectors_norm).astype(np.float32)

    k_eff = min(n_clusters, train_latents.shape[0])
    if k_eff <= 0:
        raise ValueError("No samples available to fit KMeans.")

    kmeans = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
    train_cluster_ids = kmeans.fit_predict(train_latents)
    cluster_centroids = kmeans.cluster_centers_.astype(np.float32)

    return {
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "pca": pca,
        "train_latents": train_latents,
        "train_cluster_ids": train_cluster_ids.astype(np.int64),
        "cluster_centroids": cluster_centroids,
        "n_clusters_eff": int(k_eff),
        "latent_dim_eff": int(latent_dim_eff),
    }


def _append_codebook_to_prompt(prompts, code_vectors):
    return np.concatenate([prompts, code_vectors], axis=1).astype(np.float32)


def _cluster_ids_to_onehot(cluster_ids, n_clusters):
    onehot = np.zeros((cluster_ids.shape[0], n_clusters), dtype=np.float32)
    onehot[np.arange(cluster_ids.shape[0]), cluster_ids] = 1.0
    return onehot


def _train_codebook_classifier(
    train_prompts,
    train_cluster_ids,
    hidden_dim,
    batch_size,
    epochs,
    lr,
    device,
):
    model = PromptToCodebookMLP(
        input_dim=train_prompts.shape[1],
        hidden_dim=hidden_dim,
        n_clusters=int(np.max(train_cluster_ids)) + 1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(
        torch.from_numpy(train_prompts),
        torch.from_numpy(train_cluster_ids.astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    pbar = tqdm(range(epochs), desc="Codebook clf train", leave=False)
    for _ in pbar:
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for prompts_b, cluster_ids_b in loader:
            prompts_b = prompts_b.to(device)
            cluster_ids_b = cluster_ids_b.to(device)

            logits = model(prompts_b)
            loss = loss_fn(logits, cluster_ids_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * prompts_b.shape[0]
            total_correct += int((logits.argmax(dim=1) == cluster_ids_b).sum().item())
            total_seen += int(prompts_b.shape[0])

        pbar.set_postfix(
            loss=f"{(total_loss / max(total_seen, 1)):.4f}",
            acc=f"{(total_correct / max(total_seen, 1)):.4f}",
        )
    return model


@torch.no_grad()
def _predict_cluster_ids(model, prompts, batch_size, device):
    model.eval()
    preds = []
    for start in range(0, prompts.shape[0], batch_size):
        end = min(start + batch_size, prompts.shape[0])
        prompts_b = torch.from_numpy(prompts[start:end]).to(device)
        logits = model(prompts_b)
        preds.append(logits.argmax(dim=1).cpu().numpy().astype(np.int64))
    return np.concatenate(preds, axis=0)


def _train_decoder(
    train_prompts,
    train_paths,
    train_path_lengths,
    train_interactions,
    train_path_padding_mask,
    hidden_dim,
    n_layers,
    n_heads,
    lr,
    weight_decay,
    batch_size,
    epochs,
    device,
):
    model = PathDecoder(
        prompt_dim=train_prompts.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_T=max(train_paths.shape[1] - 1, 1),
        prefix_len=4,
        pad_value=PAD_VALUE,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(
        torch.from_numpy(train_prompts),
        torch.from_numpy(train_paths),
        torch.from_numpy(train_path_lengths),
        torch.from_numpy(train_interactions),
        torch.from_numpy(train_path_padding_mask),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    pbar = tqdm(range(epochs), desc="Decoder train", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for prompts_b, paths_b, path_lengths_b, interactions_b, path_padding_mask_b in loader:
            prompts_b = prompts_b.to(device)
            paths_b = paths_b.to(device)
            path_lengths_b = path_lengths_b.to(device)
            interactions_b = interactions_b.to(device)
            path_padding_mask_b = path_padding_mask_b.to(device)

            paths_in = paths_b[:, :-1, :]
            interactions_in = interactions_b[:, :-1, :]
            paths_out = paths_b[:, 1:, :]
            interactions_out = interactions_b[:, 1:, :]

            (
                delay_pred,
                power_pred,
                phase_sin_pred,
                phase_cos_pred,
                phase_pred,
                az_sin_pred,
                az_cos_pred,
                az_pred,
                el_sin_pred,
                el_cos_pred,
                el_pred,
                path_length_pred,
                interaction_logits,
            ) = model(prompts_b, paths_in, interactions_in)

            total_loss, *_ = masked_loss(
                delay_pred,
                power_pred,
                phase_sin_pred,
                phase_cos_pred,
                phase_pred,
                az_sin_pred,
                az_cos_pred,
                az_pred,
                el_sin_pred,
                el_cos_pred,
                el_pred,
                path_length_pred,
                interaction_logits,
                paths_out,
                path_lengths_b,
                interactions_out,
                pad_value=PAD_VALUE,
                interaction_weight=0.1,
                path_padding_mask=path_padding_mask_b,
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            n_batches += 1

        pbar.set_postfix(epoch=epoch + 1, loss=f"{(epoch_loss / max(n_batches, 1)):.6f}")
    return model


@torch.no_grad()
def _generate_batched(model, prompts, batch_size, device):
    model.eval()
    generated_list = []
    pathcount_list = []

    for start in range(0, prompts.shape[0], batch_size):
        end = min(start + batch_size, prompts.shape[0])
        prompts_b = torch.from_numpy(prompts[start:end]).to(device)
        generated_b, pathcount_b, _ = generate_paths_no_env_batch(
            model,
            prompts_b,
            max_steps=MAX_GENERATE_STEPS,
        )
        generated_list.append(generated_b.numpy().astype(np.float32))
        pathcount_list.append(pathcount_b.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(generated_list, axis=0), np.concatenate(pathcount_list, axis=0)


def _compute_sequence_metrics(pred_paths, pred_path_lengths, gt_paths, gt_path_lengths):
    delay_sse = 0.0
    power_sse = 0.0
    joint_sse = 0.0
    n_steps = 0
    path_length_se = []

    for idx in range(gt_paths.shape[0]):
        n_valid = int(round(float(gt_path_lengths[idx].squeeze()) * MAX_GENERATE_STEPS))
        if n_valid <= 0:
            continue

        gt_seq = gt_paths[idx, 1:1 + n_valid, :2]
        pred_seq = pred_paths[idx, :n_valid, :2]
        err = pred_seq - gt_seq

        delay_sse += float(np.sum(err[:, 0] ** 2))
        power_sse += float(np.sum((err[:, 1] / 0.01) ** 2))
        joint_sse += float(np.sum(np.sum(err ** 2, axis=1)))
        n_steps += n_valid

        pred_len = float(np.squeeze(pred_path_lengths[idx]))
        gt_len = float(np.squeeze(gt_path_lengths[idx]))
        path_length_se.append((pred_len - gt_len) ** 2)

    if n_steps == 0:
        raise ValueError("No valid validation timesteps were available for evaluation.")

    delay_rmse = float(np.sqrt(delay_sse / n_steps))
    power_rmse = float(np.sqrt(power_sse / n_steps))
    joint_rmse = float(np.sqrt(joint_sse / n_steps))
    path_length_rmse = float(np.sqrt(np.mean(path_length_se))) if path_length_se else 0.0
    return delay_rmse, power_rmse, joint_rmse, path_length_rmse, int(n_steps)


def evaluate_scenario(
    scenario,
    n_clusters=16,
    latent_dim=8,
    sort_by="power",
    split_by="user",
    train_ratio=0.8,
    random_state=42,
    decoder_hidden_dim=128,
    decoder_layers=4,
    decoder_heads=4,
    decoder_lr=1e-3,
    decoder_weight_decay=1e-4,
    decoder_batch_size=64,
    decoder_epochs=40,
    clf_hidden_dim=128,
    clf_lr=1e-3,
    clf_batch_size=128,
    clf_epochs=20,
    oracle_val_codebook=False,
    codebook_repr="centroid",
):
    dm.download(scenario)
    dataset = dm.load(scenario)

    train_data = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by=split_by,
        sort_by=sort_by,
        train_ratio=train_ratio,
    )
    val_data = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by=split_by,
        sort_by=sort_by,
        train_ratio=train_ratio,
    )

    train = _prepare_data(train_data)
    val = _prepare_data(val_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codebook = _fit_codebook(
        train["seq_vectors"],
        latent_dim=latent_dim,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    prompt_mean, prompt_std = _fit_standardizer(train["prompts"])
    train_prompts_norm = _apply_standardizer(train["prompts"], prompt_mean, prompt_std)
    val_prompts_norm = _apply_standardizer(val["prompts"], prompt_mean, prompt_std)

    clf = _train_codebook_classifier(
        train_prompts=train_prompts_norm,
        train_cluster_ids=codebook["train_cluster_ids"],
        hidden_dim=clf_hidden_dim,
        batch_size=clf_batch_size,
        epochs=clf_epochs,
        lr=clf_lr,
        device=device,
    )

    train_pred_cluster_ids = _predict_cluster_ids(clf, train_prompts_norm, clf_batch_size, device)
    train_cluster_acc = float(np.mean(train_pred_cluster_ids == codebook["train_cluster_ids"]))
    val_pred_cluster_ids = _predict_cluster_ids(clf, val_prompts_norm, clf_batch_size, device)
    val_cluster_acc = None

    if oracle_val_codebook:
        val_seq_vectors_norm = _apply_standardizer(val["seq_vectors"], codebook["seq_mean"], codebook["seq_std"])
        val_latents = codebook["pca"].transform(val_seq_vectors_norm).astype(np.float32)
        val_dists = np.sum(
            (val_latents[:, None, :] - codebook["cluster_centroids"][None, :, :]) ** 2,
            axis=2,
        )
        val_cluster_ids = np.argmin(val_dists, axis=1).astype(np.int64)
    else:
        val_cluster_ids = val_pred_cluster_ids
        val_latents = None

    if codebook_repr == "centroid":
        train_code_vectors = codebook["cluster_centroids"][codebook["train_cluster_ids"]]
        val_code_vectors = codebook["cluster_centroids"][val_cluster_ids]
    elif codebook_repr == "onehot":
        train_code_vectors = _cluster_ids_to_onehot(codebook["train_cluster_ids"], codebook["n_clusters_eff"])
        val_code_vectors = _cluster_ids_to_onehot(val_cluster_ids, codebook["n_clusters_eff"])
    else:
        raise ValueError(f"Unsupported codebook representation: {codebook_repr}")

    train_prompts_with_code = _append_codebook_to_prompt(train_prompts_norm, train_code_vectors)
    val_prompts_with_code = _append_codebook_to_prompt(val_prompts_norm, val_code_vectors)

    decoder = _train_decoder(
        train_prompts=train_prompts_with_code,
        train_paths=train["paths"],
        train_path_lengths=train["path_lengths"],
        train_interactions=train["interactions"],
        train_path_padding_mask=train["path_padding_mask"],
        hidden_dim=decoder_hidden_dim,
        n_layers=decoder_layers,
        n_heads=decoder_heads,
        lr=decoder_lr,
        weight_decay=decoder_weight_decay,
        batch_size=decoder_batch_size,
        epochs=decoder_epochs,
        device=device,
    )

    pred_paths, pred_path_lengths = _generate_batched(
        decoder,
        val_prompts_with_code,
        batch_size=decoder_batch_size,
        device=device,
    )

    delay_rmse, power_rmse, joint_rmse, path_length_rmse, n_eval_steps = _compute_sequence_metrics(
        pred_paths,
        pred_path_lengths,
        val["paths"],
        val["path_lengths"],
    )

    return {
        "scenario": scenario,
        "n_clusters": int(codebook["n_clusters_eff"]),
        "latent_dim": int(codebook["latent_dim_eff"]),
        "codebook_repr": codebook_repr,
        "decoder_hidden_dim": int(decoder_hidden_dim),
        "decoder_layers": int(decoder_layers),
        "decoder_heads": int(decoder_heads),
        "decoder_epochs": int(decoder_epochs),
        "clf_hidden_dim": int(clf_hidden_dim),
        "clf_epochs": int(clf_epochs),
        "prompt_dim": int(train_prompts_with_code.shape[1]),
        "train_cluster_acc": train_cluster_acc,
        "val_codebook_mode": "oracle" if oracle_val_codebook else "predicted",
        "val_cluster_acc": val_cluster_acc,
        "n_train_samples": int(train["paths"].shape[0]),
        "n_val_samples": int(val["paths"].shape[0]),
        "n_eval": int(val["paths"].shape[0]),
        "n_eval_steps": int(n_eval_steps),
        "delay_rmse": delay_rmse,
        "power_rmse": power_rmse,
        "joint_rmse": joint_rmse,
        "path_length_rmse": path_length_rmse,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Latent codebook-conditioned PathDecoder. Fits a compact sequence codebook on train GT, predicts codebook id from prompt, and conditions the decoder on the predicted centroid."
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names",
    )
    parser.add_argument("--n-clusters", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--sort-by", type=str, default="power", choices=["power", "delay"])
    parser.add_argument("--split-by", type=str, default="user")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-weight-decay", type=float, default=1e-4)
    parser.add_argument("--decoder-batch-size", type=int, default=64)
    parser.add_argument("--decoder-epochs", type=int, default=40)
    parser.add_argument("--clf-hidden-dim", type=int, default=128)
    parser.add_argument("--clf-lr", type=float, default=1e-3)
    parser.add_argument("--clf-batch-size", type=int, default=128)
    parser.add_argument("--clf-epochs", type=int, default=20)
    parser.add_argument("--oracle-val-codebook", action="store_true")
    parser.add_argument("--codebook-repr", type=str, default="centroid", choices=["centroid", "onehot"])
    parser.add_argument(
        "--output-csv",
        type=str,
        default="kmeans_latent_codebook_conditioned_results.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    rows = []
    for scenario in scenarios:
        print(f"\nEvaluating scenario: {scenario}")
        row = evaluate_scenario(
            scenario=scenario,
            n_clusters=args.n_clusters,
            latent_dim=args.latent_dim,
            sort_by=args.sort_by,
            split_by=args.split_by,
            train_ratio=args.train_ratio,
            decoder_hidden_dim=args.decoder_hidden_dim,
            decoder_layers=args.decoder_layers,
            decoder_heads=args.decoder_heads,
            decoder_lr=args.decoder_lr,
            decoder_weight_decay=args.decoder_weight_decay,
            decoder_batch_size=args.decoder_batch_size,
            decoder_epochs=args.decoder_epochs,
            clf_hidden_dim=args.clf_hidden_dim,
            clf_lr=args.clf_lr,
            clf_batch_size=args.clf_batch_size,
            clf_epochs=args.clf_epochs,
            oracle_val_codebook=args.oracle_val_codebook,
            codebook_repr=args.codebook_repr,
        )
        rows.append(row)
        print(
            f"{scenario} | "
            f"clusters={row['n_clusters']}, "
            f"latent_dim={row['latent_dim']}, "
            f"train_cluster_acc={row['train_cluster_acc']:.4f}, "
            f"delay_rmse={row['delay_rmse']:.4f}, "
            f"power_rmse={row['power_rmse']:.4f}, "
            f"joint_rmse={row['joint_rmse']:.4f}, "
            f"n_eval={row['n_eval']}, "
            f"n_eval_steps={row['n_eval_steps']}"
        )

    output_path = Path(args.output_csv)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
