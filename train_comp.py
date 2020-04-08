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

def getCompNetworks(args):
  dance_enc = Dance_Enc(dim_z_dance=args.dim_z_dance, dim_z_movement=args.dim_z_movement,
                               hidden_size=args.dance_enc_hidden_size, num_layers=args.dance_enc_num_layers, bidirection=(args.dance_enc_bidirection==1))
  dance_dec = Dance_Dec(dim_z_dance=args.dim_z_dance, dim_z_movement=args.dim_z_movement,
                               hidden_size=args.dance_dec_hidden_size, num_layers=args.dance_dec_num_layers)
  audstyle_enc = Audstyle_Enc(aud_size=args.aud_style_size, dim_z=args.dim_z_dance)
  dance_reg = Dance2Style(aud_size=args.aud_style_size, dim_z_dance=args.dim_z_dance)
  danceAud_dis = DanceAud_Dis2(aud_size=28, dim_z_movement=args.dim_z_movement, length=3)
  zdance_dis = DanceAud_Dis2(aud_size=28, dim_z_movement=args.dim_z_dance, length=1)

  checkpoint2 = torch.load(args.neta_snapshot)
  neta_cls = AudioClassifier_rnn(10,30,28,cls=3)
  neta_cls.load_state_dict(checkpoint2)

  return dance_enc, dance_dec, audstyle_enc, dance_reg, danceAud_dis, zdance_dis, neta_cls

if __name__ == "__main__":
  parser = CompOptions()
  args = parser.parse()

  args.train = True

  if args.name is None:
    args.name = 'Comp'

  args.log_dir = os.path.join(args.log_dir, args.name)
  if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
  args.snapshot_dir = os.path.join(args.snapshot_dir, args.name)
  if not os.path.exists(args.snapshot_dir):
    os.mkdir(args.snapshot_dir)

  data_loader = get_loader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, dataset=args.dataset, data_dir=args.data_dir)

  #### Pretrain network from Decomp
  initp_enc, stdp_dec, movement_enc = loadDecompModel(args)

  #### Comp network
  dance_enc, dance_dec, audstyle_enc, dance_reg, danceAud_dis, zdance_dis, neta_cls = getCompNetworks(args)


  trainer = Trainer_Comp(data_loader,
                    movement_enc = movement_enc,
                    initp_enc = initp_enc,
                    stdp_dec = stdp_dec,
                    dance_enc = dance_enc,
                    dance_dec = dance_dec,
                    danceAud_dis = danceAud_dis,
                    zdance_dis = zdance_dis,
                    aud_enc=neta_cls,
                    audstyle_enc=audstyle_enc,
                    dance_reg=dance_reg,
                    args = args
                    )

  if not args.resume is None:
    ep, it = trainer.resume(args.resume, True)
  else:
    ep, it = 0, 0
  trainer.train(ep, it)

