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

from model_decomp import *
from networks import *
from options import DecompOptions
from data import get_loader

def getDecompNetworks(args):
  initp_enc = InitPose_Enc(pose_size=args.pose_size, dim_z_init=args.dim_z_init)
  initp_dec = InitPose_Dec(pose_size=args.pose_size, dim_z_init=args.dim_z_init)
  movement_enc = Movement_Enc(pose_size=args.pose_size, dim_z_movement=args.dim_z_movement, length=args.stdp_length,
                              hidden_size=args.movement_enc_hidden_size, num_layers=args.movement_enc_num_layers, bidirection=(args.movement_enc_bidirection==1))
  stdp_dec = StandardPose_Dec(pose_size=args.pose_size, dim_z_movement=args.dim_z_movement, dim_z_init=args.dim_z_init, length=args.stdp_length,
                             hidden_size=args.stdp_dec_hidden_size, num_layers=args.stdp_dec_num_layers)
  return initp_enc, initp_dec, movement_enc, stdp_dec

if __name__ == "__main__":
  parser = DecompOptions()
  args = parser.parse()

  args.train = True

  if args.name is None:
    args.name = 'Decomp'

  args.log_dir = os.path.join(args.log_dir, args.name)
  if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
  args.snapshot_dir = os.path.join(args.snapshot_dir, args.name)
  if not os.path.exists(args.snapshot_dir):
    os.mkdir(args.snapshot_dir)

  data_loader = get_loader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, dataset=args.dataset, data_dir=args.data_dir, tolerance=args.tolerance)

  initp_enc, initp_dec, movement_enc, stdp_dec = getDecompNetworks(args)

  trainer = Trainer_Decomp(data_loader,
                    initp_enc = initp_enc,
                    initp_dec = initp_dec,
                    movement_enc = movement_enc,
                    stdp_dec = stdp_dec,
                    args = args
                    )

  if not args.resume is None:
    ep, it = trainer.resume(args.resume, False)
  else:
    ep, it = 0, 0

  trainer.train(ep=ep, it=it)

