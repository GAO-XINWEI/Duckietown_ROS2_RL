"""
version goal: loop; reward; delete;
    -[v] loop check
    -[v] reward test
    -[] delete
    -[v] avg reward show
    -[v] overfitting?
"""
import torch
import os
import numpy as np

# debug
SYS_DEBUG_MODE = False
NN_DEBUG_MODE = False
SINGLE_DEBUG_MODE = True

# ==========
# Version
train_version = '05_31_sgm1_rwd2101_en2_lr1_gm90_kep2_1024'
load_version = '05_22_sgm1_rwd211_en2_lr1_gm90_kep2_1024'
#'03_28_sgm1_b"rwd4151_clp2sgm_0en2_lr2_gm90_gre0_kep2_drop0'
#'02_13_ppobuf_reward1_num3_step1_epoch1_lr2_gama90_greedy2_angle'
# (do not change)
project_name = 'Duckie_RL_00'
train_model_path = 'log/' + train_version + '/model'
load_model_path = 'log/' + load_version + '/model'
save_render_path = 'log/' + load_version + '/render'

# Training Config
BOOL_TRAINING = True
BOOL_RENDER = False
# (recover)
BOOL_LOAD_MODEL = False
# (save)
BOOL_SAVE_RENDER = False

# Training Structure
NUM_RLRUNNER = 4
NUM_WORKER = 1
MAX_BUFFER_PER_RUNNER = 1
MIN_BUFFER_LENGTH = 4
MAX_NUM_EPISODE = 1e8
SAVE_EPISODES = 100
EPSIODE_PER_KILL = 50

# RL Parameters
LEARNING_RATE = 1.e-5
BETAS = (0.9, 0.999)
GAMMA = 0.90
EGREEDY = 0
# ppo-ac
EPS_CLIP = 0.2
K_EPOCH_PPO = 1

# Env Parameters
STATE_SIZE = (30, 40)
CHANNAL = 3
ACTION_SIZE = 2
MAX_NUM_STEPS = 1024  # 1024
# Reward (fit to env)
TERMINIAL_REWARD = -200
COLLIDE_FACTOR = 100
MOVE_REWARD = 1.0
LANE_REWARD = 100

# Net Config
GRIDIANT_CLIP = 10.0
DROP_PROB = 0.0
NET_SIZE = 256
# Net Adjust (fit to env)
ACTOR_MEAN_FACTOR = 1.0
ACTOR_SIGMA_FACTOR = 1.0
CITIC_NET_FACTOR = 15
VARIANCE_BOUNDARY = 0.1
ENTROPY_FACTOR = 0.01  # 0.01

# ==========
# Town Config
DUCKIETOWN = ['Duckietown-straight_road-v0',
              'Duckietown-4way-v0',
              'Duckietown-udem1-v0',
              'Duckietown-small_loop-v0',
              'Duckietown-small_loop_cw-v0',
              'Duckietown-zigzag_dists-v0',
              'Duckietown-loop_obstacles-v0',
              'Duckietown-loop_pedestrians-v0']
TOWN = DUCKIETOWN[5]

# Device
CPU_ONLY = False
# (do not change)
TORCH_CPU = torch.device("cpu")
RUNNER_DEVICE = torch.device("cuda:0")
DRIVER_DEVICE = torch.device("cuda:0")
NUM_CPU = os.cpu_count()
NUM_GPU = int(torch.cuda.device_count()) if torch.cuda.is_available() and not CPU_ONLY else 0
CPU_PER_RUNNER = int(np.floor(NUM_CPU / NUM_RLRUNNER))
GPU_PER_RUNNER = 1.0 * NUM_GPU / (NUM_RLRUNNER + 1) if torch.cuda.is_available() and not CPU_ONLY else 0

# Cars-Interface
GAIN = 1.0
TRIM = 0.0
RADIUS = 0.0318
K = 27.0
LIMIT = 1.0 
WHEEL_DIST = 0.102
