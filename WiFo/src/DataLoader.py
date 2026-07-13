# coding=utf-8
import numpy as np
import torch as th
import json
import torch
import scipy.io
import datetime
import copy
import h5py
import hdf5storage
import random
import math
import os
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X_train):

        self.X_train = X_train

    def __len__(self):

        return self.X_train.shape[0]

    def __getitem__(self, idx):

        return self.X_train[idx].unsqueeze(0)


class MyFusionDataset(Dataset):
    def __init__(self, X_train, pathformer_features):
        if X_train.shape[0] != pathformer_features.shape[0]:
            raise ValueError(
                f"Channel samples ({X_train.shape[0]}) and PathFormer features "
                f"({pathformer_features.shape[0]}) must have the same length."
            )
        self.X_train = X_train
        self.pathformer_features = pathformer_features.float()

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, idx):
        return self.X_train[idx].unsqueeze(0), self.pathformer_features[idx]


def build_loader(data_tensor, batch_size, shuffle):
    dataset = MyDataset(data_tensor)
    return th.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        prefetch_factor=4,
    )


def build_fusion_loader(data_tensor, pathformer_features, batch_size, shuffle):
    dataset = MyFusionDataset(data_tensor, pathformer_features)
    return th.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        prefetch_factor=4,
    )


def channels_to_tensor(X_complex):
    return torch.cat((X_complex.real, X_complex.imag), dim=1).float()


def resolve_pathformer_feature_path(args, dataset, split):
    features_dir = getattr(args, "pathformer_features_dir", None) or os.path.join(
        os.path.dirname(__file__), "..", "dataset", "blessed_task_user_loc"
    )
    return os.path.abspath(
        os.path.join(features_dir, f"_{dataset}_{split}_pathformer_features.pt")
    )


def maybe_build_loader_with_pathformer_features(args, dataset, channels_tensor, shuffle, split):
    if not getattr(args, "use_pathformer_features", False):
        return build_loader(channels_tensor, args.batch_size, shuffle=shuffle)

    feature_path = resolve_pathformer_feature_path(args, dataset, split)
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"PathFormer feature cache not found for {dataset} ({split}): {feature_path}"
        )
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    feature_key = getattr(args, "pathformer_feature_key", "pathformer_features")
    if feature_key not in payload:
        raise KeyError(f"Expected '{feature_key}' in {feature_path}")
    pathformer_features = payload[feature_key].to(torch.float32)
    return build_fusion_loader(
        channels_tensor,
        pathformer_features,
        args.batch_size,
        shuffle=shuffle,
    )



def data_load_single(args, dataset): # 加载单个数据集

    folder_path_test = '../dataset/{}/X_test.mat'.format(dataset)

    X_test = hdf5storage.loadmat(folder_path_test)
    X_test_complex = torch.tensor(np.array(X_test['X_val'], dtype=complex)).unsqueeze(1)
    X_test = channels_to_tensor(X_test_complex)
    return build_loader(X_test, args.batch_size, shuffle=False)


def data_load_single_train(args, dataset):

    folder_path_train = '../dataset/{}/X_train.mat'.format(dataset)

    X_train = hdf5storage.loadmat(folder_path_train)
    X_train_complex = torch.tensor(np.array(X_train['X_train'], dtype=complex)).unsqueeze(1)
    X_train = channels_to_tensor(X_train_complex)
    return build_loader(X_train, args.batch_size, shuffle=True)

def data_load_single_mine(args, dataset): # 加载单个数据集

    data_dir = getattr(args, "data_dir", "/home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc")
    folder_path_test = os.path.join(data_dir, '_{}_val_data.pt'.format(dataset))

    X_test_complex = torch.load(folder_path_test)['channels']
    X_test = channels_to_tensor(X_test_complex)
    return maybe_build_loader_with_pathformer_features(
        args,
        dataset,
        X_test,
        shuffle=False,
        split="val",
    )


def data_load_single_train_mine(args, dataset):

    data_dir = getattr(args, "data_dir", "/home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc")
    folder_path_train = os.path.join(data_dir, '_{}_train_data.pt'.format(dataset))

    X_train_complex = torch.load(folder_path_train)['channels']
    X_train = channels_to_tensor(X_train_complex)
    return maybe_build_loader_with_pathformer_features(
        args,
        dataset,
        X_train,
        shuffle=True,
        split="train",
    )

def data_load(args):

    test_data_all = []

    for dataset_name in args.dataset.split('*'):
        test_data = data_load_single(args, dataset_name)
        test_data_all.append(test_data)
    
    return test_data_all


def data_load_main(args):

    test_data = data_load(args)

    return test_data

def data_load_mine(args):
    test_data_all = []
    for dataset_name in args.dataset.split('*'):
        test_data = data_load_single_mine(args, dataset_name)
        test_data_all.append(test_data)
    return test_data_all


def data_load_train(args):

    train_data_all = []

    for dataset_name in args.dataset.split('*'):
        train_data = data_load_single_train(args, dataset_name)
        train_data_all.append(train_data)

    return train_data_all


def data_load_train_mine(args):
    train_data_all = []
    for dataset_name in args.dataset.split('*'):
        train_data = data_load_single_train_mine(args, dataset_name)
        train_data_all.append(train_data)
    return train_data_all
