# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/Dancing2Music/License.txt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import numpy as np

if torch.cuda.is_available():
  T = torch.cuda
else:
  T = torch

###########################################################
##########
##########         Stage 1: Movement
##########
###########################################################
class InitPose_Enc(nn.Module):
  def __init__(self, pose_size, dim_z_init):
    super(InitPose_Enc, self).__init__()
    nf = 64
    #nf = 32
    self.enc = nn.Sequential(
      nn.Linear(pose_size, nf),
      nn.LayerNorm(nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, nf),
      nn.LayerNorm(nf),
      nn.LeakyReLU(0.2, inplace=True),
    )
    self.mean = nn.Sequential(
      nn.Linear(nf,dim_z_init),
    )
    self.std = nn.Sequential(
      nn.Linear(nf,dim_z_init),
    )
  def forward(self, pose):
    enc = self.enc(pose)
    return self.mean(enc), self.std(enc)

class InitPose_Dec(nn.Module):
  def __init__(self, pose_size, dim_z_init):
    super(InitPose_Dec, self).__init__()
    nf = 64
    #nf = dim_z_init
    self.dec = nn.Sequential(
      nn.Linear(dim_z_init, nf),
      nn.LayerNorm(nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, nf),
      nn.LayerNorm(nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf,pose_size),
    )
  def forward(self, z_init):
    return self.dec(z_init)

class Movement_Enc(nn.Module):
  def __init__(self, pose_size, dim_z_movement, length, hidden_size, num_layers, bidirection=False):
    super(Movement_Enc, self).__init__()
    self.hidden_size = hidden_size
    self.bidirection = bidirection
    if bidirection:
      self.num_dir = 2
    else:
      self.num_dir = 1
    self.recurrent = nn.GRU(pose_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirection)
    self.init_h = nn.Parameter(torch.randn(num_layers*self.num_dir, 1, hidden_size).type(T.FloatTensor), requires_grad=True)
    if bidirection:
      self.mean = nn.Sequential(
        nn.Linear(hidden_size*2,dim_z_movement),
      )
      self.std = nn.Sequential(
        nn.Linear(hidden_size*2,dim_z_movement),
      )
    else:
      '''
      self.enc = nn.Sequential(
        nn.Linear(hidden_size, hidden_size//2),
        nn.LayerNorm(hidden_size//2),
        nn.ReLU(inplace=True),
      )
      '''
      self.mean = nn.Sequential(
        nn.Linear(hidden_size,dim_z_movement),
      )
      self.std = nn.Sequential(
        nn.Linear(hidden_size,dim_z_movement),
      )
  def forward(self, poses):
    num_samples = poses.shape[0]
    h_t = [self.init_h.repeat(1, num_samples, 1)]
    output, hidden = self.recurrent(poses, h_t[0])
    if self.bidirection:
      output = torch.cat((output[:,-1,:self.hidden_size], output[:,0,self.hidden_size:]), 1)
    else:
      output = output[:,-1,:]
    #enc = self.enc(output)
    #return self.mean(enc), self.std(enc)
    return self.mean(output), self.std(output)

  def getFeature(self, poses):
    num_samples = poses.shape[0]
    h_t = [self.init_h.repeat(1, num_samples, 1)]
    output, hidden = self.recurrent(poses, h_t[0])
    if self.bidirection:
      output = torch.cat((output[:,-1,:self.hidden_size], output[:,0,self.hidden_size:]), 1)
    else:
      output = output[:,-1,:]
    return output

class StandardPose_Dec(nn.Module):
  def __init__(self, pose_size, dim_z_init, dim_z_movement, length, hidden_size, num_layers):
    super(StandardPose_Dec, self).__init__()
    self.length = length
    self.pose_size = pose_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    #dim_z_init=0
    '''
    self.z2init = nn.Sequential(
      nn.Linear(dim_z_init+dim_z_movement, hidden_size),
      nn.LayerNorm(hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, num_layers*hidden_size)
    )
    '''
    self.z2init = nn.Sequential(
      nn.Linear(dim_z_init+dim_z_movement, num_layers*hidden_size)
    )
    self.recurrent = nn.GRU(dim_z_movement, hidden_size, num_layers=num_layers, batch_first=True)
    self.pose_g = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.LayerNorm(hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, pose_size)
    )

  def forward(self, z_init, z_movement):
    h_init = self.z2init(torch.cat((z_init, z_movement), 1))
    #h_init = self.z2init(z_movement)
    h_init = h_init.view(self.num_layers, h_init.size(0), self.hidden_size)
    z_movements = z_movement.view(z_movement.size(0),1,z_movement.size(1)).repeat(1, self.length, 1)
    z_m_t, _ = self.recurrent(z_movements, h_init)
    z_m = z_m_t.contiguous().view(-1, self.hidden_size)
    poses = self.pose_g(z_m)
    poses = poses.view(z_movement.shape[0], self.length, self.pose_size)
    return poses

class StandardPose_Dis(nn.Module):
  def __init__(self, pose_size, length):
    super(StandardPose_Dis, self).__init__()
    self.pose_size = pose_size
    self.length = length
    nd = 1024
    self.main = nn.Sequential(
      nn.Linear(length*pose_size, nd),
      nn.LayerNorm(nd),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd,nd//2),
      nn.LayerNorm(nd//2),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd//2,nd//4),
      nn.LayerNorm(nd//4),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd//4, 1)
    )
  def forward(self, pose_seq):
    pose_seq = pose_seq.view(-1, self.pose_size*self.length)
    return self.main(pose_seq).squeeze()

###########################################################
##########
##########         Stage 2: Dance
##########
###########################################################
class Dance_Enc(nn.Module):
  def __init__(self, dim_z_movement, dim_z_dance, hidden_size, num_layers, bidirection=False):
    super(Dance_Enc, self).__init__()
    self.hidden_size = hidden_size
    self.bidirection = bidirection
    if bidirection:
      self.num_dir = 2
    else:
      self.num_dir = 1
    self.recurrent = nn.GRU(2*dim_z_movement, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirection)
    self.init_h = nn.Parameter(torch.randn(num_layers*self.num_dir, 1, hidden_size).type(T.FloatTensor), requires_grad=True)
    if bidirection:
      self.mean = nn.Sequential(
        nn.Linear(hidden_size*2,dim_z_dance),
      )
      self.std = nn.Sequential(
        nn.Linear(hidden_size*2,dim_z_dance),
      )
    else:
      self.mean = nn.Sequential(
        nn.Linear(hidden_size,dim_z_dance),
      )
      self.std = nn.Sequential(
        nn.Linear(hidden_size,dim_z_dance),
      )
  def forward(self, movements_mean, movements_std):
    movements = torch.cat((movements_mean, movements_std),2)
    num_samples = movements.shape[0]
    h_t = [self.init_h.repeat(1, num_samples, 1)]
    output, hidden = self.recurrent(movements, h_t[0])
    if self.bidirection:
      output = torch.cat((output[:,-1,:self.hidden_size], output[:,0,self.hidden_size:]), 1)
    else:
      output = output[:,-1,:]
    return self.mean(output), self.std(output)

class Dance_Dec(nn.Module):
  def __init__(self, dim_z_dance, dim_z_movement, hidden_size, num_layers):
    super(Dance_Dec, self).__init__()
    #self.length = length
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.dim_z_movement = dim_z_movement
    #dim_z_init=0
    '''
    self.z2init = nn.Sequential(
      nn.Linear(dim_z_init+dim_z_movement, hidden_size),
      nn.LayerNorm(hidden_size),
      nn.ReLU(True),
      nn.Linear(hidden_size, num_layers*hidden_size)
    )
    '''
    self.z2init = nn.Sequential(
      nn.Linear(dim_z_dance, num_layers*hidden_size)
    )
    self.recurrent = nn.GRU(dim_z_dance, hidden_size, num_layers=num_layers, batch_first=True)
    self.movement_g = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.LayerNorm(hidden_size),
      nn.ReLU(True),
      #nn.Linear(hidden_size, dim_z_movement)
    )
    self.mean = nn.Sequential(
      nn.Linear(hidden_size,dim_z_movement),
    )
    self.std = nn.Sequential(
      nn.Linear(hidden_size,dim_z_movement),
    )

  def forward(self, z_dance, length=3):
    h_init = self.z2init(z_dance)
    h_init = h_init.view(self.num_layers, h_init.size(0), self.hidden_size)
    z_dance = z_dance.view(z_dance.size(0),1,z_dance.size(1)).repeat(1, length, 1)
    z_d_t, _ = self.recurrent(z_dance, h_init)
    z_d = z_d_t.contiguous().view(-1, self.hidden_size)
    z_movement = self.movement_g(z_d)
    z_movement_mean, z_movement_std = self.mean(z_movement), self.std(z_movement)
    #z_movement = z_movement.view(z_dance.shape[0], length, self.dim_z_movement)
    return z_movement_mean, z_movement_std


class DanceAud_Dis2(nn.Module):
  def __init__(self, aud_size, dim_z_movement, length=3):
    super(DanceAud_Dis2, self).__init__()
    self.aud_size = aud_size
    self.dim_z_movement = dim_z_movement
    self.length = length
    nd = 1024
    self.movementd = nn.Sequential(
      nn.Linear(dim_z_movement*2*length, nd),
      nn.LayerNorm(nd),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd,nd//2),
      nn.LayerNorm(nd//2),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd//2,nd//4),
      nn.LayerNorm(nd//4),
      nn.LeakyReLU(0.2, inplace=True),
      #nn.Linear(nd//4, 30),
      nn.Linear(nd//4, 30),
    )

    self.audd = nn.Sequential(
      nn.Linear(aud_size, 30),
      nn.LayerNorm(30),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(30, 30),
      nn.LayerNorm(30),
      nn.LeakyReLU(0.2, inplace=True),
    )
    self.jointd = nn.Sequential(
      nn.Linear(60, 1)
    )

  def forward(self, movements, aud):
    if len(movements.shape) == 3:
      movements = movements.view(movements.shape[0], movements.shape[1]*movements.shape[2])
    m = self.movementd(movements)
    a = self.audd(aud)
    ma = torch.cat((m,a),1)

    return self.jointd(ma).squeeze(), None

class DanceAud_Dis(nn.Module):
  def __init__(self, aud_size, dim_z_movement, length=3):
    super(DanceAud_Dis, self).__init__()
    self.aud_size = aud_size
    self.dim_z_movement = dim_z_movement
    self.length = length
    nd = 1024
    self.movementd = nn.Sequential(
      #nn.Linear(dim_z_movement*3, nd),
      nn.Linear(dim_z_movement*2, nd),
      nn.LayerNorm(nd),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd,nd//2),
      nn.LayerNorm(nd//2),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd//2,nd//4),
      nn.LayerNorm(nd//4),
      nn.LeakyReLU(0.2, inplace=True),
      #nn.Linear(nd//4, 30),
      nn.Linear(nd//4, 30),
    )


  def forward(self, movements, aud):
    #movements = movements.view(movements.shape[0], movements.shape[1]*movements.shape[2])
    m = self.movementd(movements)
    return m.squeeze()
    #a = self.audd(aud)
    #ma = torch.cat((m,a),1)

    #return self.jointd(ma).squeeze()

class DanceAud_InfoDis(nn.Module):
  def __init__(self, aud_size, dim_z_movement, length):
    super(DanceAud_InfoDis, self).__init__()
    self.aud_size = aud_size
    self.dim_z_movement = dim_z_movement
    self.length = length
    nd = 1024

    self.movementd = nn.Sequential(
      nn.Linear(dim_z_movement*6, nd*2),
      nn.LayerNorm(nd*2),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd*2, nd),
      nn.LayerNorm(nd),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd,nd//2),
      nn.LayerNorm(nd//2),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd//2,nd//4),
      nn.LayerNorm(nd//4),
      nn.LeakyReLU(0.2, inplace=True),
    )

    self.dis = nn.Sequential(
      nn.Linear(nd//4, 1)
    )
    self.reg = nn.Sequential(
      nn.Linear(nd//4, aud_size)
    )

  def forward(self, movements, aud):
    movements = movements.view(movements.shape[0], movements.shape[1]*movements.shape[2])
    m = self.movementd(movements)
    return self.dis(m).squeeze(), self.reg(m)

class Dance2Style(nn.Module):
  def __init__(self, dim_z_dance, aud_size):
    super(Dance2Style, self).__init__()
    self.aud_size = aud_size
    self.dim_z_dance = dim_z_dance
    nd = 512
    self.main = nn.Sequential(
      nn.Linear(dim_z_dance, nd),
      nn.LayerNorm(nd),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd, nd//2),
      nn.LayerNorm(nd//2),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd//2, nd//4),
      nn.LayerNorm(nd//4),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nd//4, aud_size),
    )
  def forward(self, zdance):
    return self.main(zdance)

###########################################################
##########
##########         Audio
##########
###########################################################
class AudioClassifier_rnn(nn.Module):
  def __init__(self, dim_z_motion, hidden_size, pose_size, cls, num_layers=1, h_init=2):
    super(AudioClassifier_rnn, self).__init__()
    self.dim_z_motion = dim_z_motion
    self.hidden_size = hidden_size
    self.pose_size = pose_size
    self.h_init = h_init
    self.num_layers = num_layers

    self.init_h = nn.Parameter(torch.randn(1, 1, self.hidden_size).type(T.FloatTensor), requires_grad=True)
    self.recurrent = nn.GRU(pose_size, hidden_size, num_layers=num_layers, batch_first=True)
    self.classifier = nn.Sequential(
      #nn.Dropout(p=0.2),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(True),
      #nn.Dropout(p=0.2),
      nn.Linear(hidden_size, cls)
    )
  def forward(self, poses):
    hidden, _ = self.recurrent(poses, self.init_h.repeat(1, poses.shape[0], 1))
    last_hidden = hidden[:,-1,:]
    cls = self.classifier(last_hidden)
    return cls
  def get_style(self, auds):
    hidden, _ = self.recurrent(auds, self.init_h.repeat(1, auds.shape[0], 1))
    last_hidden = hidden[:,-1,:]
    return last_hidden


class Audstyle_Enc(nn.Module):
  def __init__(self, aud_size, dim_z, dim_noise=30):
    super(Audstyle_Enc, self).__init__()
    self.dim_noise = dim_noise
    nf = 64
    #nf = 32
    self.enc = nn.Sequential(
      nn.Linear(aud_size+dim_noise, nf),
      nn.LayerNorm(nf),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(nf, nf*2),
      nn.LayerNorm(nf*2),
      nn.LeakyReLU(0.2, inplace=True),
    )
    self.mean = nn.Sequential(
      nn.Linear(nf*2,dim_z),
    )
    self.std = nn.Sequential(
      nn.Linear(nf*2,dim_z),
    )
  def forward(self, aud):
    noise = torch.randn(aud.shape[0], self.dim_noise).cuda()
    y = torch.cat((aud, noise), 1)
    enc = self.enc(y)
    return self.mean(enc), self.std(enc)
