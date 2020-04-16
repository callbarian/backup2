# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d

from PIL import Image
import argparse
import os 

_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = 64
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type

  imagenet_pretrained = FLAGS.imagenet_pretrained

  NUM_CLASSES = 400
  if eval_type == 'rgb600':
    NUM_CLASSES = 600

  if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

  if eval_type == 'rgb600':
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
  else:
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))


    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)

    _, end_points = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
    end_feature = end_points['avg_pool3d']
    

    rgb_variable_map = {}
    for variable in tf.global_variables():

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      flow_logits, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb' or eval_type == 'rgb600':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)

  with tf.Session() as sess:
    feed_dict = {}
    if eval_type in ['rgb', 'rgb600', 'joint']:
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
      tf.logging.info('RGB checkpoint restored')
      



    video_list = open(VIDEO_PATH_FILE).readlines()
    video_list = [name.strip() for name in video_list]
    print('video_list', video_list)
    if not os.path.isdir(OUTPUT_FEAT_DIR):
        os.mkdir(OUTPUT_FEAT_DIR)

    print('Total number of videos: %d'%len(video_list))
    
    for cnt, video_name in enumerate(video_list):
        video_path = os.path.join(VIDEO_DIR, video_name)
        feat_path = os.path.join(OUTPUT_FEAT_DIR, video_name + '.npy')

        if os.path.exists(feat_path):
            print('Feature file for video %s already exists.'%video_name)
            continue

        print('video_path', video_path)
        
        n_frame = len([ff for ff in os.listdir(video_path) if ff.endswith('.jpg')])
        
        print('Total frames: %d'%n_frame)
        
        features = []

        n_feat = int(n_frame // 8)
        n_batch = n_feat // batch_size + 1
        print('n_frame: %d; n_feat: %d'%(n_frame, n_feat))
        print('n_batch: %d'%n_batch)

        for i in range(n_batch):
            input_blobs = []
            for j in range(batch_size):
                input_blob = []
                for k in range(L):
                    idx = i*batch_size*L + j*L + k
                    idx = int(idx)
                    idx = idx%n_frame + 1
                    image = Image.open(os.path.join(video_path, '%d.jpg'%idx))
                    image = image.resize((resize_w, resize_h))
                    image = np.array(image, dtype='float32')
                    '''
                    image[:, :, 0] -= 104.
                    image[:, :, 1] -= 117.
                    image[:, :, 2] -= 123.
                    '''
                    image[:, :, :] -= 127.5
                    image[:, :, :] /= 127.5
                    input_blob.append(image)
                
                input_blob = np.array(input_blob, dtype='float32')
                
                input_blobs.append(input_blob)

            input_blobs = np.array(input_blobs, dtype='float32')

            clip_feature = sess.run(end_feature, feed_dict={rgb_input: input_blobs})
            clip_feature = np.reshape(clip_feature, (-1, clip_feature.shape[-1]))
            
            features.append(clip_feature)

        features = np.concatenate(features, axis=0)
        features = features[:n_feat:2]   # 16 frames per feature  (since 64-frame snippet corresponds to 8 features in I3D)

        feat_path = os.path.join(OUTPUT_FEAT_DIR, video_name + '.npy')

        print('Saving features and probs for video: %s ...'%video_name)
        np.save(feat_path, features)
        
        print('%d: %s has been processed...'%(cnt, video_name))
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  print('******--------- Extract I3D features ------*******')
  parser.add_argument('-g', '--GPU', type=int, default=0, help='GPU id')
  parser.add_argument('-of', '--OUTPUT_FEAT_DIR', dest='OUTPUT_FEAT_DIR', type=str,
                        default='./feature_out/',
                        help='Output feature path')
  parser.add_argument('-vpf', '--VIDEO_PATH_FILE', type=str,
                        default='video_list.txt',
                        help='input video list')
  parser.add_argument('-vd', '--VIDEO_DIR', type=str,
                        default='./input_video/',
                        help='frame directory')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  OUTPUT_FEAT_DIR = params['OUTPUT_FEAT_DIR']
  VIDEO_PATH_FILE = params['VIDEO_PATH_FILE']
  VIDEO_DIR = params['VIDEO_DIR']
  RUN_GPU = params['GPU']

  resize_w = 224
  resize_h = 224
  L = 64
  batch_size = 1

  # set gpu id
  os.environ['CUDA_VISIBLE_DEVICES'] = str(RUN_GPU)

  tf.app.run(main)
