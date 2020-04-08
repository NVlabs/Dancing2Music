# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/Dancing2Music/License.txt

import os
import argparse
import functools

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model_comp import *
from networks import *
from options import CompOptions
from data import get_loader


if __name__ == "__main__":
  parser = CompOptions()
  args = parser.parse()
  #### Pretrain network from Decomp
  initp_enc, stdp_dec, movement_enc = loadDecompModel(args)

  #### Comp network
  dance_enc, dance_dec, audstyle_enc, dance_reg, danceAud_dis, zdance_dis, neta_cls = loadCompModel(args)

  mean_pose=np.load('../onbeat/all_onbeat_mean.npy')
  std_pose=np.load('../onbeat/all_onbeat_std.npy')
  mean_aud=np.load('../onbeat/all_aud_mean.npy')
  std_aud=np.load('../onbeat/all_aud_std.npy')


def loadDecompModel(args):
  initp_enc = InitPose_Enc(pose_size=args.pose_size, dim_z_init=args.dim_z_init)
  stdp_dec = StandardPose_Dec(pose_size=args.pose_size, dim_z_movement=args.dim_z_movement, dim_z_init=args.dim_z_init, length=args.stdp_length,
                          hidden_size=args.stdp_dec_hidden_size, num_layers=args.stdp_dec_num_layers)
  movement_enc = Movement_Enc(pose_size=args.pose_size, dim_z_movement=args.dim_z_movement, length=args.stdp_length,
                             hidden_size=args.movement_enc_hidden_size, num_layers=args.movement_enc_num_layers, bidirection=(args.movement_enc_bidirection==1))
  checkpoint = torch.load(args.decomp_snapshot)
  initp_enc.load_state_dict(checkpoint['initp_enc'])
  stdp_dec.load_state_dict(checkpoint['stdp_dec'])
  movement_enc.load_state_dict(checkpoint['movement_enc'])
  return initp_enc, stdp_dec, movement_enc

def loadCompModel(args):
  dance_enc = Dance_Enc(dim_z_dance=args.dim_z_dance, dim_z_movement=args.dim_z_movement,
                               hidden_size=args.dance_enc_hidden_size, num_layers=args.dance_enc_num_layers, bidirection=(args.dance_enc_bidirection==1))
  dance_dec = Dance_Dec(dim_z_dance=args.dim_z_dance, dim_z_movement=args.dim_z_movement,
                               hidden_size=args.dance_dec_hidden_size, num_layers=args.dance_dec_num_layers)
  audstyle_enc = Audstyle_Enc(aud_size=args.aud_style_size, dim_z=args.dim_z_dance)
  dance_reg = Dance2Style(aud_size=args.aud_style_size, dim_z_dance=args.dim_z_dance)
  danceAud_dis = DanceAud_Dis2(aud_size=28, dim_z_movement=args.dim_z_movement, length=3)
  zdance_dis = DanceAud_Dis2(aud_size=28, dim_z_movement=args.dim_z_dance, length=1)
  checkpoint = torch.load(args.resume)
  dance_enc.load_state_dict(checkpoint['dance_enc'])
  dance_dec.load_state_dict(checkpoint['dance_dec'])
  audstyle_enc.load_state_dict(checkpoint['audstyle_enc'])
  dance_reg.load_state_dict(checkpoint['dance_reg'])
  danceAud_dis.load_state_dict(checkpoint['danceAud_dis'])
  zdance_dis.load_state_dict(checkpoint['zdance_dis'])

  checkpoint2 = torch.load(args.neta_snapshot)
  neta_cls = AudioClassifier_rnn(10,30,28,cls=3)
  neta_cls.load_state_dict(checkpoint2)

  return dance_enc, dance_dec, audstyle_enc, dance_reg, danceAud_dis, zdance_dis, neta_cls
