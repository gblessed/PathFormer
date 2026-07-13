# coding=utf-8
import argparse
import random
import os
from model import WiFo_model
from train import TrainLoop


import setproctitle
import torch

from DataLoader import data_load_main, data_load_mine, data_load_train, data_load_train_mine
from utils import *

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
#num_samples x Tx x Subcarriers
#8x32 =8x[4] =8x2:
ALL_SCENARIOS = [
    "city_0_newyork_3p5_s",
    "city_1_losangeles_3p5",
    "city_2_chicago_3p5",
    "city_3_houston_3p5",
    "city_4_phoenix_3p5",
    "city_5_philadelphia_3p5",
    "city_6_miami_3p5",
    "city_7_sandiego_3p5",
    "city_8_dallas_3p5",
    "city_9_sanfrancisco_3p5",
    "city_10_austin_3p5",
    "city_11_santaclara_3p5",
    "city_12_fortworth_3p5",
    "city_13_columbus_3p5",
    "city_17_seattle_3p5_s",
    "city_18_denver_3p5",
    "city_19_oklahoma_3p5_s",
    "city_16_sanfrancisco_3p5_lwm",
    "city_23_beijing_3p5",
    "city_31_barcelona_3p5",
    "city_35_san_francisco_3p5",
    "city_47_chicago_3p5",
    "city_89_nairobi_3p5",
    "city_91_xiangyang_3p5",
    "city_92_sãopaulo_3p5",
    "boston5g_3p5",
    "city_86_ankara_3p5",
    # "city_72_capetown_3p5",
    "city_84_baoding_3p5",
    "city_95_delhi_3p5",
    "city_96_osaka_3p5",
    "city_88_tongshan_3p5",
]
def normalize_dataset_arg(dataset_arg):
    if dataset_arg in {"all", "all_scenarios"}:
        return "*".join(ALL_SCENARIOS)
    return dataset_arg


def dataset_tag(dataset_arg):
    datasets = dataset_arg.split('*')
    if dataset_arg == "*".join(ALL_SCENARIOS):
        return f"all_scenarios_{len(datasets)}"
    if len(datasets) == 1:
        return datasets[0]
    return f"multi_{len(datasets)}"


def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    return th.device("cpu")

def create_argparser():
    defaults = dict(
        # experimental settings
        note = '',
        task = 'short',
        file_load_path = '',
        dataset = 'DS1',
        used_data = '',
        process_name = 'process_name',
        his_len = 6,
        pred_len = 6,
        few_ratio = 0.5,
        stage = 0,
        do_finetune = False,
        finetune_epochs = 20,
        eval_interval = 1,
        save_every_epoch = False,

        # model settings
        mask_ratio = 0.5,
        patch_size = 4, #4,
        t_patch_size = 2,
        size = 'base',
        no_qkv_bias = 0,
        pos_emb = 'SinCos',
        conv_num = 3,

        # pretrain settings
        random=True,
        mask_strategy = 'random',
        mask_strategy_random = 'batch', # ['none','batch']
        
        # training parameters
        lr=1e-3,
        min_lr = 1e-5,
        early_stop = 10,
        weight_decay=0.05,
        batch_size=256,
        log_interval=5,
        total_epoches = 10000,
        device_id='1',
        machine = 'machine_name',
        clip_grad = 0.05,  # 0.05
        lr_anneal_steps = 200,
        my_data = True,
        data_dir = '/home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc',
        input_noise_snr_db = 20.0,
        freeze_wifo_backbone = False,
        train_decoder_pred = False,
        adapter_mode = 'none',
        use_pathformer_features = False,
        pathformer_feature_dim = 1024,
        pathformer_features_dir = '/home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc',
        pathformer_feature_key = 'pathformer_features',
        pathformer_token_steps = 25,
        context_gate_init = 0.0,
        context_fusion_position = 'after_decoder',

    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    #X_test: [1000, 1, 24, 4, 128]
    # 

    th.autograd.set_detect_anomaly(True)
    
    args = create_argparser().parse_args()
    args.use_pathformer_features = args.adapter_mode in {'pathformer', 'pathformer_tokens'}
    if args.adapter_mode == 'pathformer_tokens' and args.pathformer_feature_key == 'pathformer_features':
        args.pathformer_feature_key = 'pathformer_token_features'
    setproctitle.setproctitle("{}-{}".format(args.process_name, args.device_id))
    setup_init(100)  # 随机种子设定100
    args.dataset = normalize_dataset_arg(args.dataset)
    ##creates a list of datasets split with * e.g. D1*D2 [D1,D2]

    if args.my_data:
        train_data = data_load_train_mine(args) if args.do_finetune else None
        test_data = data_load_mine(args)
    else:
        train_data = data_load_train(args) if args.do_finetune else None
        test_data = data_load_main(args) # 加载数据

    dataset_name_for_folder = dataset_tag(args.dataset)
    args.folder = 'Dataset_{}_Task_{}_FewRatio_{}_{}_{}/'.format(dataset_name_for_folder, args.task, args.few_ratio, args.size, args.note)

    experiment_tags = []
    if args.freeze_wifo_backbone:
        experiment_tags.append("frozen")
    if args.train_decoder_pred:
        experiment_tags.append("decoderpred")
    if args.adapter_mode != "none":
        experiment_tags.append(args.adapter_mode)
    if args.input_noise_snr_db < 0:
        experiment_tags.append("nointernalnoise")

    run_prefix = 'Finetune_' if args.do_finetune else 'Test_'
    args.folder = run_prefix + args.folder
    if experiment_tags:
        args.folder = "{}_{}".format("-".join(experiment_tags), args.folder)

    if args.mask_strategy_random != 'batch':
        # freq_0.5
        args.folder = '{}_{}'.format(args.mask_strategy,args.mask_ratio) + args.folder
    args.model_path = './experiments/{}'.format(args.folder)
    logdir = "./logs/{}".format(args.folder)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        os.makedirs(args.model_path+'model_save/') 

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)
    device = dev(args.device_id)

    model = WiFo_model(args=args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')
    if args.file_load_path != '':
        model.load_state_dict(torch.load('{}.pkl'.format(args.file_load_path),map_location=device), strict=False)
        print('pretrained model loaded'+args.file_load_path)
    
    # torch.nn.init.kaiming_normal_(model.decoder_pred.weight)
    # torch.nn.init.zeros_(model.decoder_pred.bias)

    TrainLoop(
        args=args,
        writer=writer,
        model=model,
        train_data=train_data,
        test_data=test_data,
        device=device,
        early_stop=args.early_stop,
    ).run_loop()


if __name__ == "__main__":
    main()
