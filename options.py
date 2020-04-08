# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/Dancing2Music/License.txt

import argparse


class DecompOptions():
  def __init__(self):
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)

    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--snapshot_ep', type=int, default=1)
    parser.add_argument('--snapshot_dir', default='./snapshot')
    parser.add_argument('--data_dir', default='./data')

    # Model architecture
    parser.add_argument('--pose_size', type=int, default=28)
    parser.add_argument('--dim_z_init', type=int, default=10)
    parser.add_argument('--dim_z_movement', type=int, default=512)
    parser.add_argument('--stdp_length', type=int, default=32)
    parser.add_argument('--movement_enc_bidirection', type=int, default=1)
    parser.add_argument('--movement_enc_hidden_size', type=int, default=1024)
    parser.add_argument('--stdp_dec_hidden_size', type=int, default=1024)
    parser.add_argument('--movement_enc_num_layers', type=int, default=1)
    parser.add_argument('--stdp_dec_num_layers', type=int, default=1)
    # Training
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--latent_dropout', type=float, default=0.3)
    parser.add_argument('--lambda_kl', type=float, default=0.01)
    parser.add_argument('--lambda_initp_recon', type=float, default=1)
    parser.add_argument('--lambda_initp_consistency', type=float, default=1)
    parser.add_argument('--lambda_stdp_recon', type=float, default=1)
    parser.add_argument('--lambda_dist_z_movement', type=float, default=1)
    # Others
    parser.add_argument('--num_workers', type=int,  default=4)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--tolerance', action='store_true')

    self.parser = parser

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    return self.opt

class CompOptions():
  def __init__(self):
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)

    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--snapshot_ep', type=int, default=1)
    parser.add_argument('--snapshot_dir', default='./snapshot')
    parser.add_argument('--data_dir', default='./data')
    # Network architecture
    parser.add_argument('--pose_size', type=int, default=28)
    parser.add_argument('--aud_style_size', type=int, default=30)
    parser.add_argument('--dim_z_init', type=int, default=10)
    parser.add_argument('--dim_z_movement', type=int, default=512)
    parser.add_argument('--dim_z_dance', type=int, default=512)
    parser.add_argument('--stdp_length', type=int, default=32)
    parser.add_argument('--movement_enc_bidirection', type=int, default=1)
    parser.add_argument('--movement_enc_hidden_size', type=int, default=1024)
    parser.add_argument('--stdp_dec_hidden_size', type=int, default=1024)
    parser.add_argument('--movement_enc_num_layers', type=int, default=1)
    parser.add_argument('--stdp_dec_num_layers', type=int, default=1)
    parser.add_argument('--dance_enc_bidirection', type=int, default=0)
    parser.add_argument('--dance_enc_hidden_size', type=int, default=1024)
    parser.add_argument('--dance_enc_num_layers', type=int, default=1)
    parser.add_argument('--dance_dec_hidden_size', type=int, default=1024)
    parser.add_argument('--dance_dec_num_layers', type=int, default=1)
    # Training
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--latent_dropout', type=float, default=0.3)
    parser.add_argument('--lambda_kl', type=float, default=0.01)
    parser.add_argument('--lambda_kl_dance', type=float, default=0.01)
    parser.add_argument('--lambda_gan', type=float, default=1)
    parser.add_argument('--lambda_zmovements_recon', type=float, default=1)
    parser.add_argument('--lambda_stdpSeq_recon', type=float, default=10)
    parser.add_argument('--lambda_dist_z_movement', type=float, default=1)
    # Other
    parser.add_argument('--num_workers', type=int,  default=4)
    parser.add_argument('--decomp_snapshot', required=True)
    parser.add_argument('--neta_snapshot', default='./data/stats/aud_3cls.ckpt')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--dataset', type=int, default=2)
    self.parser = parser

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    return self.opt

class TestOptions():
  def __init__(self):
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)

    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--snapshot_ep', type=int, default=1)
    parser.add_argument('--snapshot_dir', default='./snapshot')
    parser.add_argument('--data_dir', default='./data')
    # Network architecture
    parser.add_argument('--pose_size', type=int, default=28)
    parser.add_argument('--aud_style_size', type=int, default=30)
    parser.add_argument('--dim_z_init', type=int, default=10)
    parser.add_argument('--dim_z_movement', type=int, default=512)
    parser.add_argument('--dim_z_dance', type=int, default=512)
    parser.add_argument('--stdp_length', type=int, default=32)
    parser.add_argument('--movement_enc_bidirection', type=int, default=1)
    parser.add_argument('--movement_enc_hidden_size', type=int, default=1024)
    parser.add_argument('--stdp_dec_hidden_size', type=int, default=1024)
    parser.add_argument('--movement_enc_num_layers', type=int, default=1)
    parser.add_argument('--stdp_dec_num_layers', type=int, default=1)
    parser.add_argument('--dance_enc_bidirection', type=int, default=0)
    parser.add_argument('--dance_enc_hidden_size', type=int, default=1024)
    parser.add_argument('--dance_enc_num_layers', type=int, default=1)
    parser.add_argument('--dance_dec_hidden_size', type=int, default=1024)
    parser.add_argument('--dance_dec_num_layers', type=int, default=1)
    # Training
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--latent_dropout', type=float, default=0.3)
    parser.add_argument('--lambda_kl', type=float, default=0.01)
    parser.add_argument('--lambda_kl_dance', type=float, default=0.01)
    parser.add_argument('--lambda_gan', type=float, default=1)
    parser.add_argument('--lambda_zmovements_recon', type=float, default=1)
    parser.add_argument('--lambda_stdpSeq_recon', type=float, default=10)
    parser.add_argument('--lambda_dist_z_movement', type=float, default=1)
    # Other
    parser.add_argument('--num_workers', type=int,  default=4)
    parser.add_argument('--decomp_snapshot', required=True)
    parser.add_argument('--comp_snapshot', required=True)
    parser.add_argument('--neta_snapshot', default='./data/stats/aud_3cls.ckpt')
    parser.add_argument('--dataset', type=int, default=2)
    parser.add_argument('--thr', type=int, default=50)
    parser.add_argument('--aud_path', type=str, required=True)
    parser.add_argument('--modulate', action='store_true')
    parser.add_argument('--out_file', type=str, default='demo/out.mp4')
    parser.add_argument('--out_dir', type=str, default='demo/out_frame')
    self.parser = parser

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    return self.opt
