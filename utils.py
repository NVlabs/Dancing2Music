# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/Dancing2Music/License.txt

import numpy as np
import pickle
import cv2
import math
import os
import random
import tensorflow as tf

class Logger(object):
  def __init__(self, log_dir):
    self.writer = tf.summary.FileWriter(log_dir)

  def scalar_summary(self, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    self.writer.add_summary(summary, step)

def vis(poses, outdir, aud=None):
  colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

  # find connection in the specified sequence, center 29 is in the position 15
  limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]

  neglect = [14,15,16,17]

  for t in range(poses.shape[0]):
    #break
    canvas = np.ones((256,500,3), np.uint8)*255

    thisPeak = poses[t]
    for i in range(18):
      if i in neglect:
        continue
      if thisPeak[i,0] == -1:
        continue
      cv2.circle(canvas, tuple(thisPeak[i,0:2].astype(int)), 4, colors[i], thickness=-1)

    for i in range(17):
      limbid = np.array(limbSeq[i])-1
      if limbid[0] in neglect or limbid[1] in neglect:
        continue
      X = thisPeak[[limbid[0],limbid[1]], 1]
      Y = thisPeak[[limbid[0],limbid[1]], 0]
      if X[0] == -1 or Y[0]==-1 or X[1]==-1 or Y[1]==-1:
        continue
      stickwidth = 4
      cur_canvas = canvas.copy()
      mX = np.mean(X)
      mY = np.mean(Y)
      #print(X, Y, limbid)
      length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
      angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
      polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
      #print(i, n, int(mY), int(mX), limbid, X, Y)
      cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
      canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    if aud is not None:
      if aud[:,t] == 1:
        cv2.circle(canvas, (30, 30), 20, (0,0,255), -1)
        #canvas = cv2.copyMakeBorder(canvas,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
    cv2.imwrite(os.path.join(outdir, 'frame{0:03d}.png'.format(t)),canvas)

def vis2(poses, outdir, fibeat):
  colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

  # find connection in the specified sequence, center 29 is in the position 15
  limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]


  neglect = [14,15,16,17]

  ibeat = cv2.imread(fibeat);
  ibeat = cv2.resize(ibeat, (500,200))

  for t in range(poses.shape[0]):
    subibeat = ibeat.copy()
    canvas = np.ones((256+200,500,3), np.uint8)*255
    canvas[256:,:,:] = subibeat

    overlay = canvas.copy()
    cv2.rectangle(overlay, (int(500/poses.shape[0]*(t+1)),256),(500,256+200), (100,100,100), -1)
    cv2.addWeighted(overlay, 0.4, canvas, 1-0.4, 0, canvas)
    thisPeak = poses[t]
    for i in range(18):
      if i in neglect:
        continue
      if thisPeak[i,0] == -1:
        continue
      cv2.circle(canvas, tuple(thisPeak[i,0:2].astype(int)), 4, colors[i], thickness=-1)

    for i in range(17):
      limbid = np.array(limbSeq[i])-1
      if limbid[0] in neglect or limbid[1] in neglect:
        continue
      X = thisPeak[[limbid[0],limbid[1]], 1]
      Y = thisPeak[[limbid[0],limbid[1]], 0]
      if X[0] == -1 or Y[0]==-1 or X[1]==-1 or Y[1]==-1:
        continue
      stickwidth = 4
      cur_canvas = canvas.copy()
      mX = np.mean(X)
      mY = np.mean(Y)
      #print(X, Y, limbid)
      length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
      angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
      polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
      #print(i, n, int(mY), int(mX), limbid, X, Y)
      cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
      canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    cv2.imwrite(os.path.join(outdir, 'frame{0:03d}.png'.format(t)),canvas)

def vis_single(pose, outfile):
  colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

  # find connection in the specified sequence, center 29 is in the position 15
  limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]

  neglect = [14,15,16,17]

  for t in range(1):
    #break
    canvas = np.ones((256,500,3), np.uint8)*255

    thisPeak = pose
    for i in range(18):
      if i in neglect:
        continue
      if thisPeak[i,0] == -1:
        continue
      cv2.circle(canvas, tuple(thisPeak[i,0:2].astype(int)), 4, colors[i], thickness=-1)

    for i in range(17):
      limbid = np.array(limbSeq[i])-1
      if limbid[0] in neglect or limbid[1] in neglect:
        continue
      X = thisPeak[[limbid[0],limbid[1]], 1]
      Y = thisPeak[[limbid[0],limbid[1]], 0]
      if X[0] == -1 or Y[0]==-1 or X[1]==-1 or Y[1]==-1:
        continue
      stickwidth = 4
      cur_canvas = canvas.copy()
      mX = np.mean(X)
      mY = np.mean(Y)
      length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
      angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
      polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
      cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
      canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    cv2.imwrite(outfile,canvas)
