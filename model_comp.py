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

class Trainer_Comp(object):
  def __init__(self, data_loader, dance_enc, dance_dec, danceAud_dis, movement_enc, initp_enc, stdp_dec, aud_enc, audstyle_enc, dance_reg=None, args=None, zdance_dis=None):
    self.data_loader = data_loader
    self.movement_enc = movement_enc
    self.initp_enc = initp_enc
    self.stdp_dec = stdp_dec
    self.dance_enc = dance_enc
    self.dance_dec = dance_dec
    self.danceAud_dis = danceAud_dis
    self.aud_enc = aud_enc
    self.audstyle_enc = audstyle_enc
    self.train = args.train
    self.args = args

    if args.train:
      self.zdance_dis = zdance_dis
      self.dance_reg = dance_reg

      self.logger = Logger(args.log_dir)
      self.logs = self.init_logs()
      self.log_interval = args.log_interval
      self.snapshot_ep = args.snapshot_ep
      self.snapshot_dir = args.snapshot_dir

      self.opt_dance_enc = torch.optim.Adam(self.dance_enc.parameters(), lr=args.lr)
      self.opt_dance_dec = torch.optim.Adam(self.dance_dec.parameters(), lr=args.lr)
      self.opt_danceAud_dis = torch.optim.Adam(self.danceAud_dis.parameters(), lr=args.lr)
      self.opt_audstyle_enc = torch.optim.Adam(self.audstyle_enc.parameters(), lr=args.lr)
      self.opt_zdance_dis = torch.optim.Adam(self.zdance_dis.parameters(), lr=args.lr)
      self.opt_dance_reg = torch.optim.Adam(self.dance_reg.parameters(), lr=args.lr)

      self.opt_stdp_dec = torch.optim.Adam(self.stdp_dec.parameters(), lr=args.lr*0.1)
      self.opt_movement_enc = torch.optim.Adam(self.movement_enc.parameters(), lr=args.lr*0.1)

    self.latent_dropout = nn.Dropout(p=args.latent_dropout)
    self.l1_criterion = torch.nn.L1Loss()
    self.gan_criterion = nn.BCEWithLogitsLoss()
    self.mse_criterion = nn.MSELoss().cuda()

  def init_logs(self):
    return {'l_kl_zdance':0, 'l_kl_zmovement':0, 'l_kl_fake_zdance':0, 'l_kl_fake_zmovement':0,
            'l_l1_zmovement_mu':0, 'l_l1_zmovement_logvar':0, 'l_l1_stdpSeq':0, 'l_l1_zdance':0,
            'l_dis':0, 'l_dis_true':0, 'l_dis_fake':0,
            'l_info':0, 'l_info_real':0, 'l_info_fake':0,
            'l_gen':0
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
  def kld_coef(self, i):
    return float(1/(1+np.exp(-0.0005*(i-15000))))


  def forward(self, stdpSeq, batchsize, aud_style, aud):
    self.aud = torch.mean(aud, dim=1)

    self.batchsize = batchsize
    self.stdpSeq = stdpSeq
    self.aud_style = aud_style
    ### stdpSeq -> z_inits, z_movements
    self.pose_0 = stdpSeq[:,0,:]
    self.z_init_mu, self.z_init_logvar = self.initp_enc(self.pose_0)
    z_init_std = self.z_init_logvar.mul(0.5).exp_()
    z_init_eps = self.get_z_random(z_init_std.size(0), z_init_std.size(1), 'gauss')
    self.z_init = z_init_eps.mul(z_init_std).add_(self.z_init_mu)

    self.z_movement_mus, self.z_movement_logvars = self.movement_enc(stdpSeq)
    z_movement_stds = self.z_movement_logvars.mul(0.5).exp_()
    z_movement_epss = self.get_z_random(z_movement_stds.size(0), z_movement_stds.size(1), 'gauss')
    self.z_movements = z_movement_epss.mul(z_movement_stds).add_(self.z_movement_mus)
    self.z_movementSeq_mu = self.z_movement_mus.view(batchsize, -1, self.z_movements.shape[1])
    self.z_movementSeq_logvar = self.z_movement_logvars.view(batchsize, -1, self.z_movements.shape[1])

    self.z_init, self.z_movements = self.z_init.detach(), self.z_movements.detach()
    self.z_movement_mus, self.z_movement_logvars = self.z_movement_mus.detach(), self.z_movement_logvars.detach()

    ### z_movements -> z_dance
    self.z_dance_mu, self.z_dance_logvar = self.dance_enc(self.z_movementSeq_mu, self.z_movementSeq_logvar)
    z_dance_std = self.z_dance_logvar.mul(0.5).exp_()
    z_dance_eps = self.get_z_random(z_dance_std.size(0), z_dance_std.size(1), 'gauss')
    self.z_dance = z_dance_eps.mul(z_dance_std).add_(self.z_dance_mu)
    ### z_dance -> z_movements
    self.recon_z_movements_mu, self.recon_z_movements_logvar  = self.dance_dec(self.z_dance)
    recon_z_movement_std = self.recon_z_movements_logvar.mul(0.5).exp_()
    recon_z_movement_eps = self.get_z_random(recon_z_movement_std.size(0), recon_z_movement_std.size(1), 'gauss')
    self.recon_z_movements = recon_z_movement_eps.mul(recon_z_movement_std).add_(self.recon_z_movements_mu)
    ### z_movements -> stdpSeq
    self.recon_stdpSeq = self.stdp_dec(self.z_init, self.recon_z_movements)

    ### Music to z_dance to z_movements
    self.fake_z_dance_mu, self.fake_z_dance_logvar = self.audstyle_enc(aud_style)
    fake_z_dance_std = self.fake_z_dance_logvar.mul(0.5).exp_()
    fake_z_dance_eps = self.get_z_random(fake_z_dance_std.size(0), fake_z_dance_std.size(1), 'gauss')
    self.fake_z_dance = fake_z_dance_eps.mul(fake_z_dance_std).add_(self.fake_z_dance_mu)
    self.fake_z_movements_mu, self.fake_z_movements_logvar = self.dance_dec(self.fake_z_dance)
    fake_z_movements_std = self.fake_z_movements_logvar.mul(0.5).exp_()
    fake_z_movements_eps = self.get_z_random(fake_z_movements_std.size(0), fake_z_movements_std.size(1), 'gauss')
    self.fake_z_movements = fake_z_movements_eps.mul(fake_z_movements_std).add_(self.fake_z_movements_mu)

    fake_z_movementSeq_mu = self.fake_z_movements_mu.view(batchsize, -1, self.fake_z_movements_mu.shape[1])
    fake_z_movementSeq_logvar = self.fake_z_movements_logvar.view(batchsize, -1, self.fake_z_movements_logvar.shape[1])
    self.fake_z_movementSeq = torch.cat((fake_z_movementSeq_mu, fake_z_movementSeq_logvar),2)

  def backward_D(self):
    #real_movements = torch.cat((self.z_movementSeq_mu, self.z_movementSeq_logvar),2)
    tmp_recon_mu = self.recon_z_movements_mu.view(self.batchsize, -1, self.z_movements.shape[1])
    tmp_recon_logvar = self.recon_z_movements_logvar.view(self.batchsize, -1, self.z_movements.shape[1])
    real_movements = torch.cat((tmp_recon_mu, tmp_recon_logvar),2)
    fake_movements = self.fake_z_movementSeq

    real_labels,_ = self.danceAud_dis(real_movements.detach(), self.aud)
    fake_labels,_ = self.danceAud_dis(fake_movements.detach(), self.aud)

    ones = self.ones_like(real_labels)
    zeros = self.zeros_like(fake_labels)

    self.loss_dis_true = self.gan_criterion(real_labels, ones)
    self.loss_dis_fake = self.gan_criterion(fake_labels, zeros)
    self.loss_dis = (self.loss_dis_true + self.loss_dis_fake)*self.args.lambda_gan

    real_dance = torch.cat((self.z_dance_mu, self.z_dance_logvar), 1)
    fake_dance = torch.cat((self.fake_z_dance_mu, self.fake_z_dance_logvar), 1)
    real_labels, _ = self.zdance_dis(real_dance.detach(), self.aud)
    fake_labels, _ = self.zdance_dis(fake_dance.detach(), self.aud)
    ones = self.ones_like(real_labels)
    zeros = self.zeros_like(fake_labels)

    self.loss_zdis_true = self.gan_criterion(real_labels, ones)
    self.loss_zdis_fake = self.gan_criterion(fake_labels, zeros)
    self.loss_dis += (self.loss_zdis_true + self.loss_zdis_fake)*self.args.lambda_gan


  def backward_danceED(self):
    # z_dance KL
    kl_element = self.z_dance_mu.pow(2).add_(self.z_dance_logvar.exp()).mul_(-1).add_(1).add_(self.z_dance_logvar)
    self.loss_kl_z_dance = torch.mean( (torch.sum(kl_element, dim=1).mul_(-0.5) * self.args.lambda_kl_dance))
    kl_element = self.fake_z_dance_mu.pow(2).add_(self.fake_z_dance_logvar.exp()).mul_(-1).add_(1).add_(self.fake_z_dance_logvar)
    self.loss_kl_fake_z_dance = torch.mean( (torch.sum(kl_element, dim=1).mul_(-0.5) * self.args.lambda_kl_dance))
    # z_movement KL
    kl_element = self.recon_z_movements_mu.pow(2).add_(self.recon_z_movements_logvar.exp()).mul_(-1).add_(1).add_(self.recon_z_movements_logvar)
    self.loss_kl_z_movement = torch.mean( (torch.sum(kl_element, dim=1).mul_(-0.5) * self.args.lambda_kl))
    kl_element = self.fake_z_movements_mu.pow(2).add_(self.fake_z_movements_logvar.exp()).mul_(-1).add_(1).add_(self.fake_z_movements_logvar)
    self.loss_kl_fake_z_movements = torch.mean( (torch.sum(kl_element, dim=1).mul_(-0.5) * self.args.lambda_kl))
    # z_movement reconstruction
    self.loss_l1_z_movement_mu = self.l1_criterion(self.recon_z_movements_mu, self.z_movement_mus) * self.args.lambda_zmovements_recon
    self.loss_l1_z_movement_logvar = self.l1_criterion(self.recon_z_movements_logvar, self.z_movement_logvars) * self.args.lambda_zmovements_recon

    # stdp reconstruction
    self.loss_l1_stdpSeq = self.l1_criterion(self.recon_stdpSeq, self.stdpSeq) * self.args.lambda_stdpSeq_recon

    # Music2Dance GAN
    fake_movements = self.fake_z_movementSeq
    fake_labels, _ = self.danceAud_dis(fake_movements, self.aud)

    ones = self.ones_like(fake_labels)
    self.loss_gen = self.gan_criterion(fake_labels, ones) * self.args.lambda_gan

    fake_dance = torch.cat((self.fake_z_dance_mu, self.fake_z_dance_logvar), 1)
    fake_labels, _ = self.zdance_dis(fake_dance, self.aud)
    ones = self.ones_like(fake_labels)
    self.loss_gen += self.gan_criterion(fake_labels, ones) * self.args.lambda_gan

    self.loss = self.loss_kl_z_movement + self.loss_kl_z_dance + self.loss_l1_z_movement_mu + self.loss_l1_z_movement_logvar + self.loss_l1_stdpSeq + self.loss_gen

  def backward_info_ondance(self):
    real_pred = self.dance_reg(self.z_dance)
    fake_pred = self.dance_reg(self.fake_z_dance)
    self.loss_info_real = self.mse_criterion(real_pred, self.aud_style)
    self.loss_info_fake = self.mse_criterion(fake_pred, self.aud_style)
    self.loss_info = self.loss_info_real + self.loss_info_fake

  def zero_grad(self, opt_list):
    for opt in opt_list:
      opt.zero_grad()

  def clip_norm(self, network_list):
    for network in network_list:
      clip_grad_norm_(network.parameters(), 0.5)

  def step(self, opt_list):
    for opt in opt_list:
      opt.step()

  def update(self):
    self.zero_grad([self.opt_danceAud_dis, self.opt_zdance_dis])
    self.backward_D()
    self.loss_dis.backward(retain_graph=True)
    self.clip_norm([self.danceAud_dis, self.zdance_dis])
    self.step([self.opt_danceAud_dis, self.opt_zdance_dis])

    self.zero_grad([self.opt_dance_enc, self.opt_dance_dec, self.opt_audstyle_enc, self.opt_stdp_dec])
    self.backward_danceED()
    self.loss.backward(retain_graph=True)
    self.clip_norm([self.dance_enc, self.dance_dec, self.audstyle_enc, self.stdp_dec])
    self.step([self.opt_dance_enc, self.opt_dance_dec, self.opt_audstyle_enc, self.opt_stdp_dec])

    self.zero_grad([self.opt_dance_enc, self.opt_audstyle_enc, self.opt_dance_reg, self.opt_stdp_dec])
    self.backward_info_ondance()
    self.loss_info.backward()
    self.clip_norm([self.dance_enc, self.audstyle_enc, self.dance_reg, self.stdp_dec])
    self.step([self.opt_dance_enc, self.opt_audstyle_enc, self.opt_dance_reg, self.opt_stdp_dec])

  def test_final(self, initpose, aud, n, thr=0):
    self.cuda()
    self.movement_enc.eval()
    self.stdp_dec.eval()
    self.initp_enc.eval()
    self.dance_enc.eval()
    self.dance_dec.eval()
    self.aud_enc.eval()
    self.audstyle_enc.eval()
    aud_style = self.aud_enc.get_style(aud).detach()

    self.fake_z_dance_mu, self.fake_z_dance_logvar = self.audstyle_enc(aud_style)
    fake_z_dance_std = self.fake_z_dance_logvar.mul(0.5).exp_()
    fake_z_dance_eps = self.get_z_random(fake_z_dance_std.size(0), fake_z_dance_std.size(1), 'gauss')
    self.fake_z_dance = fake_z_dance_eps.mul(fake_z_dance_std).add_(self.fake_z_dance_mu)

    self.fake_z_movements_mu, self.fake_z_movements_logvar = self.dance_dec(self.fake_z_dance, length=3)
    fake_z_movements_std = self.fake_z_movements_logvar.mul(0.5).exp_()
    fake_z_movements_eps = self.get_z_random(fake_z_movements_std.size(0), fake_z_movements_std.size(1), 'gauss')
    self.fake_z_movements = fake_z_movements_eps.mul(fake_z_movements_std).add_(self.fake_z_movements_mu)

    fake_stdpSeq=[]
    for i in range(n):
      z_init_mus, z_init_logvars = self.initp_enc(initpose)
      z_init_stds = z_init_logvars.mul(0.5).exp_()
      z_init_epss = self.get_z_random(z_init_stds.size(0), z_init_stds.size(1), 'gauss')
      z_init = z_init_epss.mul(z_init_stds).add_(z_init_mus)
      fake_stdp = self.stdp_dec(z_init, self.fake_z_movements[i:i+1])
      fake_stdpSeq.append(fake_stdp)
      initpose  = fake_stdp[:,-1,:]
    fake_stdpSeq = torch.cat(fake_stdpSeq, dim=0)
    flag = False
    for i in range(n):
      s = fake_stdpSeq[i]
      diff = torch.abs(s[1:]-s[:-1])
      diffsum = torch.sum(diff)
      if diffsum.cpu().detach().numpy() < thr:
        flag = True

    if flag:
      return None
    else:
      return fake_stdpSeq.cpu().detach().numpy()


  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    self.dance_enc.load_state_dict(checkpoint['dance_enc'])
    self.dance_dec.load_state_dict(checkpoint['dance_dec'])
    self.audstyle_enc.load_state_dict(checkpoint['audstyle_enc'])
    self.stdp_dec.load_state_dict(checkpoint['stdp_dec'])
    self.movement_enc.load_state_dict(checkpoint['movement_enc'])
    if train:
      self.danceAud_dis.load_state_dict(checkpoint['danceAud_dis'])
      self.dance_reg.load_state_dict(checkpoint['dance_reg'])
      self.opt_dance_enc.load_state_dict(checkpoint['opt_dance_enc'])
      self.opt_dance_dec.load_state_dict(checkpoint['opt_dance_dec'])
      self.opt_stdp_dec.load_state_dict(checkpoint['opt_stdp_dec'])
      self.opt_audstyle_enc.load_state_dict(checkpoint['opt_audstyle_enc'])
      self.opt_danceAud_dis.load_state_dict(checkpoint['opt_danceAud_dis'])
      self.opt_dance_reg.load_state_dict(checkpoint['opt_dance_reg'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    state = {
             'stdp_dec': self.stdp_dec.state_dict(),
             'movement_enc': self.movement_enc.state_dict(),
             'dance_enc': self.dance_enc.state_dict(),
             'dance_dec': self.dance_dec.state_dict(),
             'audstyle_enc': self.audstyle_enc.state_dict(),
             'danceAud_dis': self.danceAud_dis.state_dict(),
             'zdance_dis': self.zdance_dis.state_dict(),
             'dance_reg': self.dance_reg.state_dict(),
             'opt_stdp_dec': self.opt_stdp_dec.state_dict(),
             'opt_movement_enc': self.opt_movement_enc.state_dict(),
             'opt_dance_enc': self.opt_dance_enc.state_dict(),
             'opt_dance_dec': self.opt_dance_dec.state_dict(),
             'opt_audstyle_enc': self.opt_audstyle_enc.state_dict(),
             'opt_danceAud_dis': self.opt_danceAud_dis.state_dict(),
             'opt_zdance_dis': self.opt_zdance_dis.state_dict(),
             'opt_dance_reg': self.opt_dance_reg.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def cuda(self):
    if self.train:
      self.dance_reg.cuda()
      self.danceAud_dis.cuda()
      self.zdance_dis.cuda()
    self.stdp_dec.cuda()
    self.initp_enc.cuda()
    self.movement_enc.cuda()
    self.dance_enc.cuda()
    self.dance_dec.cuda()
    self.aud_enc.cuda()
    self.audstyle_enc.cuda()
    self.gan_criterion.cuda()

  def train(self, ep=0, it=0):
    self.cuda()
    for epoch in range(ep, self.args.num_epochs):
      self.movement_enc.train()
      self.stdp_dec.train()
      self.initp_enc.train()
      self.dance_enc.train()
      self.dance_dec.train()
      self.danceAud_dis.train()
      self.zdance_dis.train()
      self.audstyle_enc.train()
      self.dance_reg.train()
      self.aud_enc.eval()
      stdp_recon = 0

      for i, (stdpSeq, aud) in enumerate(self.data_loader):
        stdpSeq, aud = stdpSeq.cuda().detach(), aud.cuda().detach()
        stdpSeq = stdpSeq.view(stdpSeq.shape[0]*stdpSeq.shape[1], stdpSeq.shape[2], stdpSeq.shape[3])
        aud_style = self.aud_enc.get_style(aud).detach()

        self.forward(stdpSeq, aud.shape[0], aud_style, aud)
        self.update()
        self.logs['l_kl_zmovement'] += self.loss_kl_z_movement.data
        self.logs['l_kl_zdance'] += self.loss_kl_z_dance.data
        self.logs['l_l1_zmovement_mu'] += self.loss_l1_z_movement_mu.data
        self.logs['l_l1_zmovement_logvar'] += self.loss_l1_z_movement_logvar.data
        self.logs['l_l1_stdpSeq'] += self.loss_l1_stdpSeq.data
        self.logs['l_kl_fake_zdance'] += self.loss_kl_fake_z_dance.data
        self.logs['l_kl_fake_zmovement'] += self.loss_kl_fake_z_movements
        self.logs['l_dis'] += self.loss_dis.data
        self.logs['l_dis_true'] += self.loss_dis_true.data
        self.logs['l_dis_fake'] += self.loss_dis_fake.data
        self.logs['l_gen'] += self.loss_gen.data
        self.logs['l_info'] += self.loss_info
        self.logs['l_info_real'] += self.loss_info_real
        self.logs['l_info_fake'] += self.loss_info_fake

        print('Epoch:{:3} Iter{}/{}\tl_l1_zmovement mu{:.3f} logvar{:.3f}\tl_l1_stdpSeq {:.3f}\tl_kl_dance {:.3f}\tl_kl_movement {:.3f}\n'.format(epoch, i, len(self.data_loader),
            self.loss_l1_z_movement_mu, self.loss_l1_z_movement_logvar, self.loss_l1_stdpSeq, self.loss_kl_z_dance, self.loss_kl_z_movement) +
             '\t\t\tl_kl_f_dance {:.3f}\tl_dis {:.3f} {:.3f}\tl_gen {:.3f}'.format(self.loss_kl_fake_z_dance, self.loss_dis_true, self.loss_dis_fake, self.loss_gen))

        it += 1
        if it % self.log_interval == 0:
          for tag, value in self.logs.items():
            self.logger.scalar_summary(tag, value/self.log_interval, it)
          self.logs = self.init_logs()
      if epoch % self.snapshot_ep == 0:
        self.save(os.path.join(self.snapshot_dir, '{:04}.ckpt'.format(epoch)), epoch, it)
