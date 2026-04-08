import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from kmeans_latent_codebook_conditioned import (
    DEFAULT_SCENARIOS,
    MAX_GENERATE_STEPS,
    PAD_VALUE,
    _apply_standardizer,
    _compute_sequence_metrics,
    _fit_codebook,
    _fit_standardizer,
    _predict_cluster_ids,
    _prepare_data,
    _train_codebook_classifier,
)
from models import PathDecoder
import deepmimo as dm
from utils.utils import masked_loss


class CodebookEmbeddingPathDecoder(nn.Module):
    def __init__(
        self,
        base_prompt_dim,
        n_clusters,
        code_embed_dim,
        hidden_dim,
        n_layers,
        n_heads,
        max_T,
        prefix_len=4,
        pad_value=PAD_VALUE,
    ):
        super().__init__()
        self.code_embed = nn.Embedding(n_clusters, code_embed_dim)
        self.decoder = PathDecoder(
            prompt_dim=base_prompt_dim + code_embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_T=max_T,
            prefix_len=prefix_len,
            pad_value=pad_value,
        )

    def augment_prompts(self, prompts, code_ids):
        code_emb = self.code_embed(code_ids)
        return torch.cat([prompts, code_emb], dim=-1)

    def forward(self, prompts, paths, interactions, code_ids):
        aug_prompts = self.augment_prompts(prompts, code_ids)
        return self.decoder(aug_prompts, paths, interactions)


def _train_decoder(
    train_prompts,
    train_code_ids,
    train_paths,
    train_path_lengths,
    train_interactions,
    train_path_padding_mask,
    n_clusters,
    code_embed_dim,
    hidden_dim,
    n_layers,
    n_heads,
    lr,
    weight_decay,
    batch_size,
    epochs,
    device,
):
    model = CodebookEmbeddingPathDecoder(
        base_prompt_dim=train_prompts.shape[1],
        n_clusters=n_clusters,
        code_embed_dim=code_embed_dim,
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
        torch.from_numpy(train_code_ids.astype(np.int64)),
        torch.from_numpy(train_paths),
        torch.from_numpy(train_path_lengths),
        torch.from_numpy(train_interactions),
        torch.from_numpy(train_path_padding_mask),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    pbar = tqdm(range(epochs), desc="Code emb decoder train", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for prompts_b, code_ids_b, paths_b, path_lengths_b, interactions_b, path_padding_mask_b in loader:
            prompts_b = prompts_b.to(device)
            code_ids_b = code_ids_b.to(device)
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
            ) = model(prompts_b, paths_in, interactions_in, code_ids_b)

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
def _generate_batched(model, prompts, code_ids, batch_size, device):
    model.eval()
    generated_list = []
    pathcount_list = []

    for start in range(0, prompts.shape[0], batch_size):
        end = min(start + batch_size, prompts.shape[0])
        prompts_b = torch.from_numpy(prompts[start:end]).to(device)
        code_ids_b = torch.from_numpy(code_ids[start:end].astype(np.int64)).to(device)

        cur = torch.zeros(prompts_b.shape[0], 1, 5, device=device)
        inter_str = -1 * torch.ones(prompts_b.shape[0], 1, 4, device=device)

        outputs = []
        outputs_inter_str = []
        pathcounts = None

        for _ in range(MAX_GENERATE_STEPS):
            d, p, _, _, ph, _, _, az, _, _, el, pathcounts, inter_logits = model(
                prompts_b,
                cur,
                inter_str,
                code_ids_b,
            )

            d_t = d[:, -1]
            p_t = p[:, -1]
            ph_t = ph[:, -1]
            az_t = az[:, -1]
            el_t = el[:, -1]
            inter_logits_t = inter_logits[:, -1]
            inter_pred_t = (torch.sigmoid(inter_logits_t) > 0.5).float()

            next_path = torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1)
            outputs.append(next_path)
            outputs_inter_str.append(inter_pred_t)

            cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
            inter_str = torch.cat([inter_str, inter_pred_t.unsqueeze(1)], dim=1)

        generated_list.append(torch.stack(outputs, dim=1).detach().cpu().numpy().astype(np.float32))
        pathcount_list.append(pathcounts.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(generated_list, axis=0), np.concatenate(pathcount_list, axis=0)


def evaluate_scenario(
    scenario,
    n_clusters=8,
    latent_dim=4,
    code_embed_dim=16,
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
    decoder_epochs=20,
    clf_hidden_dim=128,
    clf_lr=1e-3,
    clf_batch_size=128,
    clf_epochs=10,
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
    val_pred_cluster_ids = _predict_cluster_ids(clf, val_prompts_norm, clf_batch_size, device)
    train_cluster_acc = float(np.mean(train_pred_cluster_ids == codebook["train_cluster_ids"]))

    decoder = _train_decoder(
        train_prompts=train_prompts_norm,
        train_code_ids=codebook["train_cluster_ids"],
        train_paths=train["paths"],
        train_path_lengths=train["path_lengths"],
        train_interactions=train["interactions"],
        train_path_padding_mask=train["path_padding_mask"],
        n_clusters=codebook["n_clusters_eff"],
        code_embed_dim=code_embed_dim,
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
        val_prompts_norm,
        val_pred_cluster_ids,
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
        "code_embed_dim": int(code_embed_dim),
        "train_cluster_acc": train_cluster_acc,
        "decoder_hidden_dim": int(decoder_hidden_dim),
        "decoder_layers": int(decoder_layers),
        "decoder_heads": int(decoder_heads),
        "decoder_epochs": int(decoder_epochs),
        "clf_hidden_dim": int(clf_hidden_dim),
        "clf_epochs": int(clf_epochs),
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
        description="Latent KMeans codebook with a learned discrete codeword embedding for PathDecoder."
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names",
    )
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--code-embed-dim", type=int, default=16)
    parser.add_argument("--sort-by", type=str, default="power", choices=["power", "delay"])
    parser.add_argument("--split-by", type=str, default="user")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-weight-decay", type=float, default=1e-4)
    parser.add_argument("--decoder-batch-size", type=int, default=64)
    parser.add_argument("--decoder-epochs", type=int, default=20)
    parser.add_argument("--clf-hidden-dim", type=int, default=128)
    parser.add_argument("--clf-lr", type=float, default=1e-3)
    parser.add_argument("--clf-batch-size", type=int, default=128)
    parser.add_argument("--clf-epochs", type=int, default=10)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="kmeans_latent_codebook_embedding_results.csv",
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
            code_embed_dim=args.code_embed_dim,
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
        )
        rows.append(row)
        print(
            f"{scenario} | "
            f"clusters={row['n_clusters']}, "
            f"latent_dim={row['latent_dim']}, "
            f"code_embed_dim={row['code_embed_dim']}, "
            f"train_cluster_acc={row['train_cluster_acc']:.4f}, "
            f"delay_rmse={row['delay_rmse']:.4f}, "
            f"power_rmse={row['power_rmse']:.4f}, "
            f"joint_rmse={row['joint_rmse']:.4f}, "
            f"n_eval={row['n_eval']}"
        )

    output_path = Path(args.output_csv)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
