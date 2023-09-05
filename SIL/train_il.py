# =====================================================================
# Date  : 2 Aug 2023
# Title : train imitation learning agent
# Creator : Iman Sharifi
# =====================================================================

from decimal import Decimal
import pandas as pd
import ast
# import csv
# import pygame
# import sys
import os
# from pygame.locals import *
from libs.IL import ILAgent
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torch.utils.data as data_utils
import time
from libs.utils import desired_action_list, get_most_recent_file
from pyswip import Prolog
import warnings

warnings.filterwarnings('ignore')
prolog = Prolog()

# Adjustable parameters ============================================
TRAIN = True  # 1 for train and 0 for test
N_EPOCHS = 10  # each epoch is 420 meters
SAVE_DATA_STEPS = int(N_EPOCHS/5)  # save weights after this number of steps
OUTPUT_EXCEL_FILE = 'excel/il_data_'

WEIGHTS_PATH1 = 'weights/lane_change/'
WEIGHTS_PATH2 = 'weights/velocity/'

# ==================================================================
timestr = time.strftime("%Y%m%d-%H%M%S")

# =====================================================================
# main
if __name__ == '__main__':

    data = pd.read_excel('excel/state_action_pair.xlsx')
    states = data['state']
    actions = data['action']
    data_length = len(states)
    train_len = int(0.9*data_length)

    # preprocessing data
    states = [ast.literal_eval(state) for state in states]
    states1 = [list(map(abs, state[:9])) for state in states]
    # states2 = [[state[0],state[4],abs(state[8]),state[9],state[13]] for state in states]
    states2 = [[state[0],state[4],state[9],state[13]] for state in states]

    actions = [ast.literal_eval(action) for action in actions]
    lane_change_actions = [desired_action_list(action[0]) for action in actions] 
    velocity_actions = [abs(action[1]) for action in actions]
    actions1, actions2 = lane_change_actions, velocity_actions

    # train-test split =================================================================
    # states
    states1_train, states1_test = states1[:train_len], states1[train_len:]
    states2_train, states2_test = states2[:train_len], states2[train_len:]

    # actions
    actions1_train, actions1_test = actions1[:train_len], actions1[train_len:]
    actions2_train, actions2_test = actions2[:train_len], actions2[train_len:]

    # convert lists to tensors =========================================================
    # train data
    states1_train = torch.tensor(states1_train, dtype=torch.float32)
    actions1_train = torch.tensor(actions1_train, dtype=torch.float32)

    states2_train = torch.tensor(states2_train, dtype=torch.float32)
    actions2_train = torch.tensor(actions2_train, dtype=torch.float32)

    # test data
    states1_test = torch.tensor(states1_test, dtype=torch.float32)
    actions1_test = torch.tensor(actions1_test, dtype=torch.float32)

    states2_test = torch.tensor(states2_test, dtype=torch.float32)
    actions2_test = torch.tensor(actions2_test, dtype=torch.float32)

    # prepare train dataset
    train_dataset1 = data_utils.TensorDataset(states1_train, actions1_train)
    train_dataset2 = data_utils.TensorDataset(states2_train, actions2_train)

    # prepare test dataset
    test_dataset1 = data_utils.TensorDataset(states1_test, actions1_test)
    test_dataset2 = data_utils.TensorDataset(states2_test, actions2_test) 

    # load dataset
    train_loader1 = data_utils.DataLoader(train_dataset1, batch_size=256, shuffle=True) 
    train_loader2 = data_utils.DataLoader(train_dataset2, batch_size=256, shuffle=True) 

    test_loader1 = data_utils.DataLoader(test_dataset1, batch_size=256, shuffle=True) 
    test_loader2 = data_utils.DataLoader(test_dataset2, batch_size=256, shuffle=True) 

    # ===============================================================
    # initialize the AgentCar object
    lc_agent = ILAgent(state_size=len(states1_train[0]), 
                       n_actions=len(actions1_train[0]), 
                       fc1_units=128, fc2_units=128, lr=0.0001, 
                       weight_file_path=WEIGHTS_PATH1)
    
    velocity_agent = ILAgent(state_size=len(states2_train[0]), n_actions=1, 
                             fc1_units=128, fc2_units=128, lr=0.001, 
                             weight_file_path=WEIGHTS_PATH2)
    
    epoch_loss_lc, epoch_loss_v = [], []

    print('start training ...')
    # the main training loop =======================================
    if TRAIN:
        batch_loss_lc, batch_loss_v = [], []

        for epoch in range(N_EPOCHS):
            for states_batch, actions_batch in train_loader1:
                lc_agent.learn(states_batch, actions_batch, softmax=True)
                batch_loss_lc.append(lc_agent.losses[-1])

            for states_batch, actions_batch in train_loader2:
                velocity_agent.learn(states_batch, actions_batch, softmax=False)
                batch_loss_v.append(velocity_agent.losses[-1])

            # print important info
            print(f'Epoch:{epoch+1}/{N_EPOCHS}')

            # save losses and weights
            if epoch % SAVE_DATA_STEPS == 0:
                # print('saving weights and data ...')
                lc_agent.save_weights(WEIGHTS_PATH1+'Weights_'+str(epoch)+'_'+str(timestr))
                velocity_agent.save_weights(WEIGHTS_PATH2+'Weights_'+str(epoch)+'_'+str(timestr))

                # dataframe for losses
                df = pd.DataFrame()
                df['lc_loss'] = batch_loss_lc
                df['v_loss'] = batch_loss_v
                df.to_excel(OUTPUT_EXCEL_FILE+'train.xlsx')
    
    # test models
    else:
        BEST_WEIGHT_LC = get_most_recent_file(dir=WEIGHTS_PATH1)
        BEST_WEIGHT_VELOCITY = get_most_recent_file(dir=WEIGHTS_PATH2)

        # load weights
        lc_agent.load_weights(WEIGHTS_PATH1+BEST_WEIGHT_LC)
        velocity_agent.load_weights(WEIGHTS_PATH2+BEST_WEIGHT_VELOCITY)

        batch_loss_lc, batch_loss_v = [], []

        for epoch in range(int(N_EPOCHS)):
            for state, desired_action in test_loader1:
                action = lc_agent.forward(state, softmax=True)
                batch_loss_lc.append(lc_agent.loss(action, desired_action).item())

            for state, desired_action in test_loader2:
                action = velocity_agent.forward(state, softmax=False)
                batch_loss_v.append(velocity_agent.loss(action, desired_action).item())

            # print important info
            print(f'Epoch:{epoch+1}/{N_EPOCHS}')

            # save losses and weights
            if epoch % SAVE_DATA_STEPS == 0:
                df = pd.DataFrame()
                df['lc_loss'] = batch_loss_lc
                df['v_loss'] = batch_loss_v
                df.to_excel(OUTPUT_EXCEL_FILE+'test.xlsx')


