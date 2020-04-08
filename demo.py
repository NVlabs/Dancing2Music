# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/Dancing2Music/License.txt
import os
import argparse
import functools
import librosa
import shutil
import sys 
sys.path.insert(0, 'preprocess')
import preprocess as p
import subprocess as sp
from shutil import copyfile

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model_comp import *
from networks import *
from options import TestOptions
import modulate
import utils

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
  checkpoint = torch.load(args.comp_snapshot)
  dance_enc.load_state_dict(checkpoint['dance_enc'])
  dance_dec.load_state_dict(checkpoint['dance_dec'])
  audstyle_enc.load_state_dict(checkpoint['audstyle_enc'])

  checkpoint2 = torch.load(args.neta_snapshot)
  neta_cls = AudioClassifier_rnn(10,30,28,cls=3)
  neta_cls.load_state_dict(checkpoint2)

  return dance_enc, dance_dec, audstyle_enc, dance_reg, danceAud_dis, zdance_dis, neta_cls

if __name__ == "__main__":
  parser = TestOptions()
  args = parser.parse()
  args.train = False

  thr = args.thr

  # Process music and get feature
  infile = args.aud_path
  outfile = 'style.npy'
  p.preprocess(infile, outfile)

  y, sr = librosa.load(infile)
  onset_env = librosa.onset.onset_strength(y, sr=sr,aggregate=np.median)
  times = librosa.frames_to_time(np.arange(len(onset_env)),sr=sr, hop_length=512)
  tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,sr=sr)
  np.save('beats.npy', times[beats])
  beats = np.round(librosa.frames_to_time(beats, sr=sr)*15)

  beats = np.load('beats.npy')
  aud = np.load('style.npy')
  os.remove('beats.npy')
  os.remove('style.npy')
  shutil.rmtree('normalized')

  #### Pretrain network from Decomp
  initp_enc, stdp_dec, movement_enc = loadDecompModel(args)

  #### Comp network
  dance_enc, dance_dec, audstyle_enc, dance_reg, danceAud_dis, zdance_dis, neta_cls = loadCompModel(args)

  trainer = Trainer_Comp(data_loader=None,
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

  print('Loading Done')

  mean_pose=np.load('{}/stats/all_onbeat_mean.npy'.format(args.data_dir))
  std_pose=np.load('{}/stats/all_onbeat_std.npy'.format(args.data_dir))
  mean_aud=np.load('{}/stats/all_aud_mean.npy'.format(args.data_dir))
  std_aud=np.load('{}/stats/all_aud_std.npy'.format(args.data_dir))


  length = aud.shape[0]

  initpose = np.zeros((14, 2))
  initpose = initpose.reshape(-1)
  #initpose = (initpose-mean_pose)/std_pose

  for j in range(aud.shape[0]):
    aud[j] = (aud[j]-mean_aud)/std_aud

  total_t = int(length/32+1)
  final_stdpSeq = np.zeros((total_t*3*32, 14, 2))
  initpose, aud = torch.Tensor(initpose).cuda(), torch.Tensor(aud).cuda()
  initpose, aud = initpose.view(1, initpose.shape[0]), aud.view(1, aud.shape[0], aud.shape[1])
  for t in range(total_t):
    print('process {}/{}'.format(t, total_t))
    fake_stdpSeq = trainer.test_final(initpose, aud, 3, thr)
    while True:
      fake_stdpSeq = trainer.test_final(initpose, aud, 3, thr)
      if not fake_stdpSeq is None:
        break
    initpose = fake_stdpSeq[2,-1]
    initpose = torch.Tensor(initpose).cuda()
    initpose = initpose.view(1,-1)
    fake_stdpSeq = fake_stdpSeq.squeeze()
    for j in range(fake_stdpSeq.shape[0]):
      for k in range(fake_stdpSeq.shape[1]):
        fake_stdpSeq[j,k] = fake_stdpSeq[j,k]*std_pose + mean_pose
    fake_stdpSeq = np.resize(fake_stdpSeq, (fake_stdpSeq.shape[0],32, 14, 2))
    for j in range(3):
      final_stdpSeq[96*t+32*j:96*t+32*(j+1)] = fake_stdpSeq[j]

  if args.modulate:
    final_stdpSeq = modulate.modulate(final_stdpSeq, beats, length)

  out_dir = args.out_dir
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  utils.vis(final_stdpSeq, out_dir)
  sp.call('ffmpeg -r 15  -i {}/frame%03d.png -i {} -c:v libx264 -pix_fmt yuv420p  -crf 23 -r 30  -y -strict -2  {}'.format(out_dir, args.aud_path, args.out_file), shell=True)

