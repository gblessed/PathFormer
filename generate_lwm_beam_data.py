import deepmimo as dm
import torch
import argparse
import numpy as np
from collections import defaultdict
from utils.utils import ChannelParameters, compute_single_array_response_torch
DEFAULT_SAVE = "/home/blessedg/Pathformer/lwm-competition-2025/blessed_task/"



def compute_beam_label_from_channel(H, S):
    if not torch.is_tensor(H):
        H = torch.from_numpy(H)
    if not torch.is_tensor(S):
        S = torch.from_numpy(S)
    H = H.to(torch.complex64)
    S = S.to(torch.complex64)
    Y = S.conj().T @ H
    prx = torch.sum(torch.abs(Y) ** 2, dim=2)
    best = torch.argmax(prx, dim=1)
    return best, prx


def make_dft_codebook(B=8):
    params = ChannelParameters()
    az_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
    el_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
    az_new = []
    el_new = []
    for az in az_t:
        for el in el_t:
            az_new.append(az)
            el_new.append(el)
    az_new = torch.tensor(az_new).unsqueeze(1)
    el_new = torch.tensor(el_new).unsqueeze(1)
    array_response = compute_single_array_response_torch(params.bs_antenna, az_new, el_new)
    return array_response.squeeze(2).T


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="city_47_chicago_3p5")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    
    
    return parser.parse_args()


def main():
    args = parse_args()
    scenario = args.scenario
    save_dir = args.save_dir
    train_ratio = args.train_ratio
    seed = args.seed
    dataset = dm.load(scenario)
    dataset.compute_channels()
    channels = dataset.channels
    # S = make_azimuth_codebook(n_beams=64)
    S = make_dft_codebook()
    if isinstance(dataset.n_ue, int):
            dataset = [dataset]
            channels = [channels]

    train_channels = []
    train_labels =[]
    test_channels = []
    test_labels = []
    for tx in range(len(dataset)):
        channel_tx = channels[tx]
        n_ue = dataset[tx].n_ue
        indices = np.arange(n_ue)
        np.random.seed(seed + tx)
        np.random.shuffle(indices)
        split_idx = int(train_ratio * len(indices))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        use_indices = dataset[tx].los != -1
        train_indices = [i for i in train_indices if use_indices[i]]
        test_indices = [i for i in test_indices if use_indices[i]]

        labels,_ = compute_beam_label_from_channel(channel_tx.squeeze(1), S)
        channel_tx = channel_tx * 1e6
        train_channel = channel_tx[train_indices]
        test_channel = channel_tx[test_indices]
        train_label = labels[train_indices]
        test_label = labels[test_indices]
        # data_dict["channels"] = channel_tx
        # data_dict["labels"] = labels

        train_channels.append(train_channel)
        train_labels.append(train_label)

        test_channels.append(test_channel)
        test_labels.append(test_label)
    data_dict  = {}
    data_dict["channels"] = torch.from_numpy(np.concatenate(train_channels)).squeeze(-1)
    data_dict["labels"] = torch.from_numpy(np.concatenate(train_labels))
    torch.save(data_dict, save_dir+f"_{scenario}_train_data.pt")
    print(f'Saved channel to {data_dict["channels"].shape} Max Label: {data_dict["labels"].max()} {save_dir+f"_{scenario}_train_data.pt"} ')


    data_dict  = {}
    data_dict["channels"] = torch.from_numpy(np.concatenate(test_channels)).squeeze(-1)
    data_dict["labels"] = torch.from_numpy(np.concatenate(test_labels))
    torch.save(data_dict, save_dir+f"_{scenario}_val_data.pt")
    print(f'Saved Validation channel {data_dict["channels"].shape} Max Label: {data_dict["labels"].max()} to {save_dir+f"_{scenario}_val_data.pt"} ')



    # print(labels.shape, channels.shape, S.shape)



    #parameters = get_parameters(num_ant_hor, num_ant_vert, n_subcarriers)
    #data.compute_channels(parameters)
    #* 1e6

if __name__ == '__main__':
    main()
