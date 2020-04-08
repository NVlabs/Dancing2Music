# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/Dancing2Music/License.txt

import os
import numpy as np
import librosa
import utils


def modulate(dance, beats, length):
  sec_interframe = 1/15

  beats_frame = np.around(beats)
  t_beat = beats_frame.astype(int)
  s_beat = np.arange(3,dance.shape[0],8)
  final_pose = np.zeros((length, 14, 2))

  if t_beat[0] >3:
    final_pose[t_beat[0]-3:t_beat[0]] = dance[:3]
  else:
    final_pose[:t_beat[0]] = dance[:t_beat[0]]
  if t_beat[0]-3 > 0:
    final_pose[:t_beat[0]-3] = dance[0]
  for t in range(t_beat.shape[0]-1):
    begin = int(t_beat[t])
    end = int(t_beat[t+1])
    interval = end-begin
    if t==s_beat.shape[0]-1:
      rest = min(final_pose.shape[0]-begin-1, dance.shape[0]-s_beat[t]-1)
      break
    if t+1 < s_beat.shape[0] and s_beat[t+1]<dance.shape[0]:
      pose = get_pose(dance[s_beat[t]:s_beat[t+1]+1], interval+1)
      if t==0 and begin>=3:
        final_pose[begin-s_beat[t]:begin] = dance[:s_beat[t]]
      final_pose[begin:end+1]=pose
      rest = min(final_pose.shape[0]-end-1, dance.shape[0]-s_beat[t+1]-1)
    else:
      end = begin
      if t+1 < s_beat.shape[0]:
        rest = min(final_pose.shape[0]-end-1, dance.shape[0]-s_beat[t+1]-1)
      else:
        print(t_beat.shape, s_beat.shape, t)
        rest = min(final_pose.shape[0]-end-1, dance.shape[0]-s_beat[t]-1)
  if rest > 0:
    if t+1 < s_beat.shape[0]:
      final_pose[end+1:end+1+rest] = dance[s_beat[t+1]+1:s_beat[t+1]+1+rest]
    else:
      final_pose[end+1:end+1+rest] = dance[s_beat[t]+1:s_beat[t]+1+rest]

  return final_pose

def get_pose(pose, n):
  t_pose = np.zeros((n, 14, 2))
  if n==11:
    t_pose[0] = pose[0]
    t_pose[1] = (pose[0]*1+pose[1]*4)/5
    t_pose[2] = (pose[1]*2+pose[2]*3)/5
    t_pose[3] = (pose[2]*3+pose[3]*2)/5
    t_pose[4] = (pose[3]*4+pose[4]*1)/5
    t_pose[5] = pose[4]
    t_pose[6] = (pose[4]*1+pose[5]*4)/5
    t_pose[7] = (pose[5]*2+pose[6]*3)/5
    t_pose[8] = (pose[6]*3+pose[7]*2)/5
    t_pose[9] = (pose[7]*4+pose[8]*1)/5
    t_pose[10] = pose[8]
  elif n==10:
    t_pose[0] = pose[0]
    t_pose[1] = (pose[0]*1+pose[1]*8)/9
    t_pose[2] = (pose[1]*2+pose[2]*7)/9
    t_pose[3] = (pose[2]*3+pose[3]*6)/9
    t_pose[4] = (pose[3]*4+pose[4]*5)/9
    t_pose[5] = (pose[4]*5+pose[5]*4)/9
    t_pose[6] = (pose[5]*6+pose[6]*3)/9
    t_pose[7] = (pose[6]*7+pose[7]*2)/9
    t_pose[8] = (pose[7]*8+pose[8]*1)/9
    t_pose[9] = pose[8]
  elif n==12:
    t_pose[0] = pose[0]
    t_pose[1] = (pose[0]*3+pose[1]*8)/11
    t_pose[2] = (pose[1]*6+pose[2]*5)/11
    t_pose[3] = (pose[2]*9+pose[3]*2)/11
    t_pose[4] = (pose[2]*1+pose[3]*10)/11
    t_pose[5] = (pose[3]*4+pose[4]*7)/11
    t_pose[6] = (pose[4]*7+pose[5]*4)/11
    t_pose[7] = (pose[5]*10+pose[6]*1)/11
    t_pose[8] = (pose[5]*2+pose[6]*9)/11
    t_pose[9] = (pose[6]*5+pose[7]*6)/11
    t_pose[10] = (pose[7]*8+pose[8]*3)/11
    t_pose[11] = pose[8]
  elif n==13:
    t_pose[0] = pose[0]
    t_pose[1] = (pose[0]*1+pose[1]*2)/3
    t_pose[2] = (pose[1]*2+pose[2]*1)/3
    t_pose[3] = pose[2]
    t_pose[4] = (pose[2]*1+pose[3]*2)/3
    t_pose[5] = (pose[3]*2+pose[4]*1)/3
    t_pose[6] = pose[4]
    t_pose[7] = (pose[4]*1+pose[5]*2)/3
    t_pose[8] = (pose[5]*2+pose[6]*1)/3
    t_pose[9] = pose[6]
    t_pose[10] = (pose[6]*1+pose[7]*2)/3
    t_pose[11] = (pose[7]*2+pose[8]*1)/3
    t_pose[12] = pose[8]
  elif n==14:
    t_pose[0] = pose[0]
    t_pose[1] = (pose[0]*5+pose[1]*8)/13
    t_pose[2] = (pose[1]*10+pose[2]*3)/13
    t_pose[3] = (pose[1]*2+pose[2]*11)/13
    t_pose[4] = (pose[2]*7+pose[3]*6)/13
    t_pose[5] = (pose[3]*12+pose[4]*1)/13
    t_pose[6] = (pose[3]*4+pose[4]*9)/13
    t_pose[7] = (pose[4]*9+pose[5]*4)/13
    t_pose[8] = (pose[4]*12+pose[5]*1)/13
    t_pose[9] = (pose[5]*6+pose[6]*7)/13
    t_pose[10] = (pose[6]*11+pose[7]*2)/13
    t_pose[11] = (pose[6]*3+pose[7]*10)/13
    t_pose[12] = (pose[7]*8+pose[8]*5)/13
    t_pose[13] = pose[8]
  elif n==9:
    t_pose = pose
  elif n==8:
    t_pose[0] = pose[0]
    t_pose[1] = (pose[1]*6+pose[2]*1)/7
    t_pose[2] = (pose[2]*5+pose[3]*2)/7
    t_pose[3] = (pose[3]*4+pose[4]*3)/7
    t_pose[4] = (pose[4]*3+pose[5]*4)/7
    t_pose[5] = (pose[5]*2+pose[6]*5)/7
    t_pose[6] = (pose[6]*1+pose[7]*6)/7
    t_pose[7] = pose[8]
  elif n==7:
    t_pose[0] = pose[0]
    t_pose[1] = (pose[1]*2+pose[2]*1)/3
    t_pose[2] = (pose[2]*1+pose[3]*2)/3
    t_pose[3] = pose[4]
    t_pose[4] = (pose[5]*2+pose[6]*1)/3
    t_pose[5] = (pose[6]*1+pose[7]*2)/3
    t_pose[6] = pose[8]
  elif n==6:
    t_pose[0] = pose[0]
    t_pose[1] = (pose[1]*2+pose[2]*3)/5
    t_pose[2] = (pose[3]*4+pose[4]*1)/5
    t_pose[3] = (pose[4]*1+pose[5]*4)/5
    t_pose[4] = (pose[6]*3+pose[7]*2)/5
    t_pose[5] = pose[8]
  elif n<6:
    t_pose[0] = pose[0]
    t_pose[n-1] = pose[8]
    for i in range(1,n-1):
      t_pose[i] = pose[4]
  elif n>14:
    t_pose[0] = pose[0]
    t_pose[n-1] = pose[8]
    for i in range(1, n-1):
      k = int(8/(n-1)*i)
      t_pose[i] = t_pose[k]
  else:
    print('NOT IMPLEMENT {}'.format(n))

  return t_pose
