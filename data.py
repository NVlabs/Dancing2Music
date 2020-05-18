# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/Dancing2Music/License.txt
import os  
import pickle
import numpy as np
import random
import torch.utils.data
from torchvision.datasets import ImageFolder
import utils


class PoseDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, tolerance=False):
    self.data_dir = data_dir
    z_fname = '{}/unitList/zumba_unit.txt'.format(data_dir)
    b_fname = '{}/unitList/ballet_unit.txt'.format(data_dir)
    h_fname = '{}/unitList/hiphop_unit.txt'.format(data_dir)
    self.z_data = []
    self.b_data = []
    self.h_data = []
    with open(z_fname, 'r') as f:
      for line in f:
        self.z_data.append([s for s in line.strip().split(' ')])
    with open(b_fname, 'r') as f:
      for line in f:
        self.b_data.append([s for s in line.strip().split(' ')])
    with open(h_fname, 'r') as f:
      for line in f:
        self.h_data.append([s for s in line.strip().split(' ')])
    self.data = [self.z_data, self.b_data, self.h_data]

    self.tolerance = tolerance
    if self.tolerance:
      z3_fname = '{}/unitList/zumba_unitseq3.txt'.format(data_dir)
      b3_fname = '{}/unitList/ballet_unitseq3.txt'.format(data_dir)
      h3_fname = '{}/unitList/hiphop_unitseq3.txt'.format(data_dir)
      z4_fname = '{}/unitList/zumba_unitseq4.txt'.format(data_dir)
      b4_fname = '{}/unitList/ballet_unitseq4.txt'.format(data_dir)
      h4_fname = '{}/unitList/hiphop_unitseq4.txt'.format(data_dir)
      z3_data = []; b3_data = []; h3_data = []; z4_data = []; b4_data = []; h4_data = []
      with open(z3_fname, 'r') as f:
        for line in f:
          z3_data.append([s for s in line.strip().split(' ')])
      with open(b3_fname, 'r') as f:
        for line in f:
          b3_data.append([s for s in line.strip().split(' ')])
      with open(h3_fname, 'r') as f:
        for line in f:
          h3_data.append([s for s in line.strip().split(' ')])
      with open(z4_fname, 'r') as f:
        for line in f:
          z4_data.append([s for s in line.strip().split(' ')])
      with open(b4_fname, 'r') as f:
        for line in f:
          b4_data.append([s for s in line.strip().split(' ')])
      with open(h4_fname, 'r') as f:
        for line in f:
          h4_data.append([s for s in line.strip().split(' ')])
      self.zt_data = z3_data + z4_data
      self.bt_data = b3_data + b4_data
      self.ht_data = h3_data + h4_data
      self.t_data = [self.zt_data, self.bt_data, self.ht_data]

    self.mean_pose=np.load(data_dir+'/stats/all_onbeat_mean.npy')
    self.std_pose=np.load(data_dir+'/stats/all_onbeat_std.npy')

  def __getitem__(self, index):
    cls = random.randint(0,2)
    cls = random.randint(0,1)
    if self.tolerance and random.randint(0,9)==0:
      index = random.randint(0, len(self.t_data[cls])-1)
      path = self.t_data[cls][index][0]
      path = os.path.join(self.data_dir, path[5:])
      orig_poses = np.load(path)
      sel = random.randint(0, orig_poses.shape[0]-1)
      orig_poses = orig_poses[sel]
    else:
      index = random.randint(0, len(self.data[cls])-1)
      path = self.data[cls][index][0]
      path = os.path.join(self.data_dir, path[5:])
      orig_poses = np.load(path)

    xjit = np.random.uniform(low=-50, high=50)
    yjit = np.random.uniform(low=-20, high=20)
    poses = orig_poses.copy()
    poses[:,:,0] += xjit
    poses[:,:,1] += yjit
    xjit = np.random.uniform(low=-50, high=50)
    yjit = np.random.uniform(low=-20, high=20)
    poses2 = orig_poses.copy()
    poses2[:,:,0] += xjit
    poses2[:,:,1] += yjit

    poses = poses.reshape(poses.shape[0], poses.shape[1]*poses.shape[2])
    poses2 = poses2.reshape(poses2.shape[0], poses2.shape[1]*poses2.shape[2])
    for i in range(poses.shape[0]):
      poses[i] = (poses[i]-self.mean_pose)/self.std_pose
      poses2[i] = (poses2[i]-self.mean_pose)/self.std_pose

    return torch.Tensor(poses), torch.Tensor(poses2)

  def __len__(self):
    return len(self.z_data)+len(self.b_data)


class MovementAudDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir):
    self.data_dir = data_dir
    z3_fname = '{}/unitList/zumba_unitseq3.txt'.format(data_dir)
    b3_fname = '{}/unitList/ballet_unitseq3.txt'.format(data_dir)
    h3_fname = '{}/unitList/hiphop_unitseq3.txt'.format(data_dir)
    z4_fname = '{}/unitList/zumba_unitseq4.txt'.format(data_dir)
    b4_fname = '{}/unitList/ballet_unitseq4.txt'.format(data_dir)
    h4_fname = '{}/unitList/hiphop_unitseq4.txt'.format(data_dir)
    self.z3_data = []
    self.b3_data = []
    self.h3_data = []
    self.z4_data = []
    self.b4_data = []
    self.h4_data = []
    with open(z3_fname, 'r') as f:
      for line in f:
        self.z3_data.append([s for s in line.strip().split(' ')])
    with open(b3_fname, 'r') as f:
      for line in f:
        self.b3_data.append([s for s in line.strip().split(' ')])
    with open(h3_fname, 'r') as f:
      for line in f:
        self.h3_data.append([s for s in line.strip().split(' ')])
    with open(z4_fname, 'r') as f:
      for line in f:
        self.z4_data.append([s for s in line.strip().split(' ')])
    with open(b4_fname, 'r') as f:
      for line in f:
        self.b4_data.append([s for s in line.strip().split(' ')])
    with open(h4_fname, 'r') as f:
      for line in f:
        self.h4_data.append([s for s in line.strip().split(' ')])
    self.data_3 = [self.z3_data, self.b3_data, self.h3_data]
    self.data_4 = [self.z4_data, self.b4_data, self.h4_data]

    z_data_root = 'zumba/'
    b_data_root = 'ballet/'
    h_data_root = 'hiphop/'
    self.data_root = [z_data_root, b_data_root, h_data_root ]
    self.mean_pose=np.load(data_dir+'/stats/all_onbeat_mean.npy')
    self.std_pose=np.load(data_dir+'/stats/all_onbeat_std.npy')
    self.mean_aud=np.load(data_dir+'/stats/all_aud_mean.npy')
    self.std_aud=np.load(data_dir+'/stats/all_aud_std.npy')

  def __getitem__(self, index):
    cls = random.randint(0,2)
    cls = random.randint(0,1)
    isthree = random.randint(0,1)

    if isthree == 0:
      index = random.randint(0, len(self.data_4[cls])-1)
      path = self.data_4[cls][index][0]
    else:
      index = random.randint(0, len(self.data_3[cls])-1)
      path = self.data_3[cls][index][0]
    path = os.path.join(self.data_dir, path[5:])
    stdpSeq = np.load(path)
    vid, cid = path.split('/')[-4], path.split('/')[-3]
    #vid, cid = vid_cid[:11], vid_cid[12:]
    aud = np.load('{}/{}/{}/{}/aud/c{}_fps15.npy'.format(self.data_dir, self.data_root[cls], vid, cid, cid))

    stdpSeq = stdpSeq.reshape(stdpSeq.shape[0], stdpSeq.shape[1], stdpSeq.shape[2]*stdpSeq.shape[3])
    for i in range(stdpSeq.shape[0]):
      for j in range(stdpSeq.shape[1]):
        stdpSeq[i,j] = (stdpSeq[i,j]-self.mean_pose)/self.std_pose
    if isthree == 0:
      start = random.randint(0,1)
      stdpSeq = stdpSeq[start:start+3]

    for i in range(aud.shape[0]):
      aud[i] = (aud[i]-self.mean_aud)/self.std_aud
    aud = aud[:30]
    return torch.Tensor(stdpSeq), torch.Tensor(aud)

  def __len__(self):
    return len(self.z3_data)+len(self.b3_data)+len(self.z4_data)+len(self.b4_data)+len(self.h3_data)+len(self.h4_data)

def get_loader(batch_size, shuffle, num_workers, dataset, data_dir, tolerance=False):
  if dataset == 0:
    a2d = PoseDataset(data_dir, tolerance)
  elif dataset == 2:
    a2d = MovementAudDataset(data_dir)
  data_loader = torch.utils.data.DataLoader(dataset=a2d,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            )
  return data_loader
