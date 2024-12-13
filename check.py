import sys, os, time
import wandb
import glob
from tqdm import tqdm

sys.path.append('./model')

from model.musemorphose import MuseMorphose
from dataloader import REMIFullSongTransformerDataset
from torch.utils.data import DataLoader

from utils import pickle_load
from torch import nn, optim
import torch
import numpy as np

import yaml
config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
trained_steps = config['training']['trained_steps']
lr_decay_steps = config['training']['lr_decay_steps']
lr_warmup_steps = config['training']['lr_warmup_steps']
no_kl_steps = config['training']['no_kl_steps']
kl_cycle_steps = config['training']['kl_cycle_steps']
kl_max_beta = config['training']['kl_max_beta']
free_bit_lambda = config['training']['free_bit_lambda']
max_lr, min_lr = config['training']['max_lr'], config['training']['min_lr']

ckpt_dir = config['training']['ckpt_dir']
params_dir = os.path.join(ckpt_dir, 'params/')
optim_dir = os.path.join(ckpt_dir, 'optim/')
pretrained_params_path = config['model']['pretrained_params_path']
pretrained_optim_path = config['model']['pretrained_optim_path']
ckpt_interval = config['training']['ckpt_interval']
log_interval = config['training']['log_interval']
val_interval = config['training']['val_interval']
constant_kl = config['training']['constant_kl']

if __name__ == "__main__":

    dset = REMIFullSongTransformerDataset(
    config['data']['data_dir'], config['data']['vocab_path'], 
    do_augment=config['training']['do_argument'], 
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['data']['dec_seqlen'], 
    model_max_bars=config['data']['max_bars'],
    pieces=pickle_load(config['data']['train_split']),
    pad_to_same=True
    )
    dset_val = REMIFullSongTransformerDataset(
    config['data']['data_dir'], config['data']['vocab_path'], 
    do_augment=config['training']['do_argument'], 
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['data']['dec_seqlen'], 
    model_max_bars=config['data']['max_bars'],
    pieces=pickle_load(config['data']['val_split']),
    pad_to_same=True
    )
    print ('[info]', '# training samples:', len(dset.pieces))

    dloader = DataLoader(dset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)
    dloader_val = DataLoader(dset_val, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)

    print("check train")
    for _ in dloader:
        pass

    print("check valid")
    for _ in dloader_val:
        pass