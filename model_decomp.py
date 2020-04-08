# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/Dancing2Music/License.txt
import os
import time
import numpy as np
import random
import math

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from utils import Logger

if torch.cuda.is_available():
  T = torch.cuda
else:
  T = torch

class Trainer_Decomp(object):
  def __init__(self, data_loader, initp_enc, initp_dec, movement_enc, stdp_dec, args=None):
    self.data_loader = data_loader
    self.initp_enc = initp_enc
    self.initp_dec = initp_dec
    self.movement_enc = movement_enc
    self.stdp_dec = stdp_dec


    self.args = args
    if args.train:
      self.logger = Logger(args.log_dir)
      self.logs = self.init_logs()
      self.log_interval = args.log_interval
      self.snapshot_ep = args.snapshot_ep
      self.snapshot_dir = args.snapshot_dir

      self.opt_initp_enc = torch.optim.Adam(self.initp_enc.parameters(), lr=args.lr)
      self.opt_initp_dec = torch.optim.Adam(self.initp_dec.parameters(), lr=args.lr)
      self.opt_movement_enc = torch.optim.Adam(self.movement_enc.parameters(), lr=args.lr)
      self.opt_stdp_dec = torch.optim.Adam(self.stdp_dec.parameters(), lr=args.lr)

    self.latent_dropout = nn.Dropout(p=args.latent_dropout)
    self.l1_criterion = torch.nn.L1Loss()
    self.gan_criterion = nn.BCEWithLogitsLoss()


  def init_logs(self):
    return {'l_kl_zinit':0, 'l_kl_zmovement':0, 'l_l1_stdp':0, 'l_l1_cross_stdp':0, 'l_dist_zmovement':0,
            'l_l1_initp':0, 'l_l1_initp_con':0,
            'kld_coef':0
            }

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda()
    return z

  @staticmethod
  def ones_like(tensor, val=1.):
    return T.FloatTensor(tensor.size()).fill_(val)

  @staticmethod
  def zeros_like(tensor, val=0.):
    return T.FloatTensor(tensor.size()).fill_(val)


  def random_generate_stdp(self, init_p):
    self.pose_0 = init_p
    self.z_init_mu, self.z_init_logvar = self.initp_enc(self.pose_0)
    z_init_std = self.z_init_logvar.mul(0.5).exp_()
    z_init_eps = self.get_z_random(z_init_std.size(0), z_init_std.size(1), 'gauss')
    self.z_init = z_init_eps.mul(z_init_std).add_(self.z_init_mu)
    self.z_random_movement = self.get_z_random(self.z_init.size(0), 512, 'gauss')
    self.fake_stdpose = self.stdp_dec(self.z_init, self.z_random_movement)
    return self.fake_stdpose

  def forward(self, stdpose1, stdpose2):
    self.stdpose1 = stdpose1
    self.stdpose2 = stdpose2

    # stdpose -> stdpose[0] -> z_init
    self.pose1_0 = stdpose1[:,0,:]
    self.pose2_0 = stdpose2[:,0,:]
    self.poses_0 = torch.cat((self.pose1_0, self.pose2_0), 0)
    self.z_init_mus, self.z_init_logvars = self.initp_enc(self.poses_0)
    z_init_stds = self.z_init_logvars.mul(0.5).exp_()
    z_init_epss = self.get_z_random(z_init_stds.size(0), z_init_stds.size(1), 'gauss')
    self.z_inits = z_init_epss.mul(z_init_stds).add_(self.z_init_mus)
    self.z_init1, self.z_init2 = torch.split(self.z_inits, self.stdpose1.size(0), dim=0)

    # stdpose -> z_movement
    stdposes = torch.cat((stdpose1, stdpose2), 0)
    self.z_movement_mus, self.z_movement_logvars = self.movement_enc(stdposes)
    z_movement_stds = self.z_movement_logvars.mul(0.5).exp_()
    z_movement_epss = self.get_z_random(z_movement_stds.size(0), z_movement_stds.size(1), 'gauss')
    self.z_movements = z_movement_epss.mul(z_movement_stds).add_(self.z_movement_mus)
    self.z_movement1, self.z_movement2 = torch.split(self.z_movements, self.stdpose1.size(0), dim=0)

    # zinit1+zmovement1->stdpose1   zinit2+zmovement2->stdpose2
    self.recon_stdpose1 = self.stdp_dec(self.z_init1, self.z_movement1)
    self.recon_stdpose2 = self.stdp_dec(self.z_init2, self.z_movement2)

    # zinit1+zmovement2->stdpose1   zinit2+zmovement1->stdpose2
    self.recon_stdpose1_cross = self.stdp_dec(self.z_init1, self.z_movement2)
    self.recon_stdpose2_cross = self.stdp_dec(self.z_init2, self.z_movement1)

    # z_init -> \hat{stdpose[0]}
    self.recon_pose1_0 = self.initp_dec(self.z_init1)
    self.recon_pose2_0 = self.initp_dec(self.z_init2)

    # single pose reconstruction
    randomlist = np.random.permutation(31)[:4]
    singlepose = []
    for r in randomlist:
      singlepose.append(self.stdpose1[:,r,:])
    self.singleposes = torch.cat(singlepose, dim=0).detach()
    self.z_single_mus, self.z_single_logvars = self.initp_enc(self.singleposes)
    z_single_stds = self.z_single_logvars.mul(0.5).exp_()
    z_single_epss = self.get_z_random(z_single_stds.size(0), z_single_stds.size(1), 'gauss')
    z_single = z_single_epss.mul(z_single_stds).add_(self.z_single_mus)
    self.recon_singleposes = self.initp_dec(z_single)

  def backward_initp_ED(self):
    # z_init KL
    kl_element = self.z_init_mus.pow(2).add_(self.z_init_logvars.exp()).mul_(-1).add_(1).add_(self.z_init_logvars)
    self.loss_kl_z_init = torch.mean( (torch.sum(kl_element, dim=1).mul_(-0.5) * self.args.lambda_kl))

    # initpose reconstruction
    self.loss_l1_initp = self.l1_criterion(self.recon_singleposes, self.singleposes) * self.args.lambda_initp_recon

    self.loss_initp = self.loss_kl_z_init + self.loss_l1_initp

  def backward_movement_ED(self):
    # z_movement KL
    kl_element = self.z_movement_mus.pow(2).add_(self.z_movement_logvars.exp()).mul_(-1).add_(1).add_(self.z_movement_logvars)
    #self.loss_kl_z_movement = torch.mean(kl_element).mul_(-0.5) * self.args.lambda_kl
    self.loss_kl_z_movement = torch.mean( (torch.sum(kl_element, dim=1).mul_(-0.5) * self.args.lambda_kl))

    # stdpose self reconstruction
    loss_l1_stdp1 = self.l1_criterion(self.recon_stdpose1, self.stdpose1) * self.args.lambda_stdp_recon
    loss_l1_stdp2 = self.l1_criterion(self.recon_stdpose2, self.stdpose2) * self.args.lambda_stdp_recon
    self.loss_l1_stdp = loss_l1_stdp1 + loss_l1_stdp2

    # stdpose cross reconstruction
    loss_l1_cross_stdp1 = self.l1_criterion(self.recon_stdpose1_cross, self.stdpose1) * self.args.lambda_stdp_recon
    loss_l1_cross_stdp2 = self.l1_criterion(self.recon_stdpose2_cross, self.stdpose2) * self.args.lambda_stdp_recon
    self.loss_l1_cross_stdp = loss_l1_cross_stdp1 + loss_l1_cross_stdp2

    # Movement dist
    self.loss_dist_z_movement = torch.mean(torch.abs(self.z_movement1-self.z_movement2)) * self.args.lambda_dist_z_movement

    self.loss_movement = self.loss_kl_z_movement + self.loss_l1_stdp + self.loss_l1_cross_stdp + self.loss_dist_z_movement


  def update(self):
    self.opt_initp_enc.zero_grad()
    self.opt_initp_dec.zero_grad()
    self.opt_movement_enc.zero_grad()
    self.opt_stdp_dec.zero_grad()
    self.backward_initp_ED()
    self.backward_movement_ED()
    self.g_loss = self.loss_initp + self.loss_movement
    self.g_loss.backward(retain_graph=True)
    clip_grad_norm_(self.movement_enc.parameters(), 0.5)
    clip_grad_norm_(self.stdp_dec.parameters(), 0.5)
    self.opt_initp_enc.step()
    self.opt_initp_dec.step()
    self.opt_movement_enc.step()
    self.opt_stdp_dec.step()


  def save(self, filename, ep, total_it):
    state = {
             'stdp_dec': self.stdp_dec.state_dict(),
             'movement_enc': self.movement_enc.state_dict(),
             'initp_enc': self.initp_enc.state_dict(),
             'initp_dec': self.initp_dec.state_dict(),
             'opt_stdp_dec': self.opt_stdp_dec.state_dict(),
             'opt_movement_enc': self.opt_movement_enc.state_dict(),
             'opt_initp_enc': self.opt_initp_enc.state_dict(),
             'opt_initp_dec': self.opt_initp_dec.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    self.stdp_dec.load_state_dict(checkpoint['stdp_dec'])
    self.movement_enc.load_state_dict(checkpoint['movement_enc'])
    self.initp_enc.load_state_dict(checkpoint['initp_enc'])
    self.initp_dec.load_state_dict(checkpoint['initp_dec'])
    # optimizer
    if train:
      self.opt_stdp_dec.load_state_dict(checkpoint['opt_stdp_dec'])
      self.opt_movement_enc.load_state_dict(checkpoint['opt_movement_enc'])
      self.opt_initp_enc.load_state_dict(checkpoint['opt_initp_enc'])
      self.opt_initp_dec.load_state_dict(checkpoint['opt_initp_dec'])
    return checkpoint['ep'], checkpoint['total_it']

  def kld_coef(self, i):
    return float(1/(1+np.exp(-0.0005*(i-15000)))) #v3


  def generate_stdp_sequence(self, initpose, aud, num_stdp):
    self.initp_enc.cuda()
    self.initp_dec.cuda()
    self.movement_enc.cuda()
    self.stdp_dec.cuda()
    self.initp_enc.eval()
    self.initp_dec.eval()
    self.movement_enc.eval()
    self.stdp_dec.eval()
    initpose = initpose.cuda()

    aud_style = self.aud_enc.get_style(aud)

    stdp_seq = []
    cnt = 0
    #for i in range(num_stdp):
    while not cnt == num_stdp:
      if cnt==0:
        z_inits = self.get_z_random(1, 10, 'gauss')
      else:
        z_init_mus, z_init_logvars = self.initp_enc(initpose)
        z_init_stds = z_init_logvars.mul(0.5).exp_()
        z_init_epss = self.get_z_random(z_init_stds.size(0), z_init_stds.size(1), 'gauss')
        z_inits = z_init_epss.mul(z_init_stds).add_(z_init_mus)

      z_audstyle_mu, z_audstyle_logvar = self.audstyle_enc(aud_style)
      z_as_std = z_audstyle_logvar.mul(0.5).exp_()
      z_as_eps = self.get_z_random(z_as_std.size(0), z_as_std.size(1), 'gauss')
      z_audstyle = z_as_eps.mul(z_as_std).add_(z_audstyle_mu)
      if random.randint(0,5)==100:
        z_audstyle = self.get_z_random(z_inits.size(0), 512, 'gauss')

      fake_stdpose = self.stdp_dec(z_inits, z_audstyle)

      s = fake_stdpose[0]
      diff = torch.abs(s[1:]-s[:-1])
      diffsum = torch.sum(diff)
      if diffsum.cpu().detach().numpy() < 70:
        continue

      cnt += 1
      stdp_seq.append(fake_stdpose.cpu().detach().numpy())
      initpose = fake_stdpose[:,-1,:]
    return stdp_seq


  def cuda(self):
    self.initp_enc.cuda()
    self.initp_dec.cuda()
    self.movement_enc.cuda()
    self.stdp_dec.cuda()
    self.l1_criterion.cuda()

  def train(self, ep=0, it=0):
    self.cuda()

    full_kl = self.args.lambda_kl
    kl_w = 0
    kl_step = 0.05
    best_stdp_recon = 100
    for epoch in range(ep, self.args.num_epochs):
      self.initp_enc.train()
      self.initp_dec.train()
      self.movement_enc.train()
      self.stdp_dec.train()
      stdp_recon = 0
      for i, (stdpose, stdpose2) in enumerate(self.data_loader):
        self.args.lambda_kl = full_kl*self.kld_coef(it)
        stdpose, stdpose2  = stdpose.cuda().detach(), stdpose2.cuda().detach()

        self.forward(stdpose, stdpose2)
        self.update()
        self.logs['l_kl_zinit'] += self.loss_kl_z_init.data
        self.logs['l_kl_zmovement'] += self.loss_kl_z_movement.data
        self.logs['l_l1_initp'] += self.loss_l1_initp.data
        self.logs['l_l1_stdp'] += self.loss_l1_stdp.data
        self.logs['l_l1_cross_stdp'] += self.loss_l1_cross_stdp.data
        self.logs['l_dist_zmovement'] += self.loss_dist_z_movement.data
        self.logs['kld_coef'] += self.args.lambda_kl

        print('Epoch:{:3} Iter{}/{}\tl_l1_initp {:.3f}\tl_l1_stdp {:.3f}\tl_l1_cross_stdp {:.3f}\tl_dist_zmove {:.3f}\tl_kl_zinit {:.3f}\t l_kl_zmove {:.3f}'.format(
              epoch, i, len(self.data_loader), self.loss_l1_initp, self.loss_l1_stdp, self.loss_l1_cross_stdp, self.loss_dist_z_movement, self.loss_kl_z_init, self.loss_kl_z_movement))

        it += 1
        if it % self.log_interval == 0:
          for tag, value in self.logs.items():
            self.logger.scalar_summary(tag, value/self.log_interval, it)
          self.logs = self.init_logs()
      if epoch % self.snapshot_ep == 0:
        self.save(os.path.join(self.snapshot_dir, '{:04}.ckpt'.format(epoch)), epoch, it)
