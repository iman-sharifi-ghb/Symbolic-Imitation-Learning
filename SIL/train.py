# =====================================================================
# Date  : 20 May 2023
# Title : train
# Creator : Iman Sharifi
# =====================================================================

from decimal import Decimal
import pandas as pd
import csv
import pygame
import sys
import os
from pygame.locals import *
from libs.agent import AgentCar
from libs.IL import ILAgent
import matplotlib.pyplot as plt
import math
import numpy as np
import time
from libs.utils import get_distance, get_lane, get_most_recent_file
from libs.utils import data_extractor
from pyswip import Prolog
import warnings

warnings.filterwarnings('ignore')

prolog = Prolog()

# Adjustable parameters ============================================
SIL = True  # True for Symbolic Imitation Learning (SIL) and False for IL only
TRAIN = True  # True for train and False for test
N_EPISODES = 30  # number of episodes for training
N_EPOCHS = 5  # each epoch is 420 meters
SAVE_DATA_STEPS = 5  # save weights after this number of steps
DIRECTION = 1  # highway direction [1 for left to right & -1 for right to left]
BEST_WEIGHT_FILE = 'weights/weights_20230524-223147'
WEIGHT_FILE = 'weights/weights'
DATASET_DIRECTORY = '../dataset/'
if SIL:
    OUTPUT_EXCEL_FILE = 'excel/sil_data.xlsx'
else:
    OUTPUT_EXCEL_FILE = 'excel/il_data.xlsx'

WEIGHTS_PATH1 = 'weights/lane_change/'
WEIGHTS_PATH2 = 'weights/velocity/'

# ==================================================================
RADAR_RENGE = 70 # PIXEL
LANES_Y = [10, 25, 36, 48, 81, 93, 108, 120]
timestr = time.strftime("%Y%m%d-%H%M%S")

# =====================================================================
# main
if __name__ == '__main__':

    # the dimentional parameters of the real-world highway and the scaled highway in pygame
    road_length, road_width, X, Y = 420, 36.12, 1366, 118
    timestep = 1 / 25

    # ===============================================================
    # initialize the AgentCar object
    agent = AgentCar(width=15, height=8, direction=DIRECTION, symbolic=SIL)

    if not SIL:
        # load lane change and velocity trained weights
        BEST_WEIGHT_LC = get_most_recent_file(dir=WEIGHTS_PATH1)
        # BEST_WEIGHT_VELOCITY = get_most_recent_file(dir=WEIGHTS_PATH2)

        # load weights
        agent.lc_agent.load_weights(WEIGHTS_PATH1+BEST_WEIGHT_LC)
        # agent.velocity_agent.load_weights(WEIGHTS_PATH2+BEST_WEIGHT_VELOCITY)

    # initialize data-extractor 
    extractor = data_extractor(directory=DATASET_DIRECTORY)

    # Pygame Settings ===============================================
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    pygame.init()  # initialize pygame
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((X, Y))

    # Load the background image here. Make sure the file exists!
    bg = pygame.image.load(DATASET_DIRECTORY + "highway.jpg")
    bg = pygame.transform.scale(bg, (X, Y))
    pygame.mouse.set_visible(1)
    pygame.display.set_caption('Highway')

    # create list for rewards
    n_hits, n_lane_changes = [], []
    traveled_distances = []
    steps, Steps = 0, []

    # episode counter
    episode = 0
    best_score = -math.inf
    max_distance = N_EPISODES * road_length

    # frame counter
    frame = 0
    print('start training ...')

    # the main training loop =======================================
    while episode < N_EPISODES:

        steps += 1

        # get vehicle info
        vehicles_id = extractor.get_ids(frame)
        vehicles_lane = extractor.get_lanes(frame)
        vehicles_direction = extractor.get_directions(frame)
        vehicles_pos = extractor.get_positions(frame)
        vehicles_vel = extractor.get_velocities(frame)
        vehicles_acc = extractor.get_accelerations(frame)
        vehicle_pos_pygame = [pygame.Rect(rect) for rect in vehicles_pos]
        neighboring_vehicles_pos = []
        neighboring_vehicles_pos_pygame = []
        target_vehicle_info = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        screen.blit(bg, (0, 0))
        x_ego, y_ego = agent.x, agent.y

        with open('prolog/vehicles_info.pl', 'w') as f:
            for vId, vLane, v_pos, v_pos_pygame, v_vel in zip(vehicles_id, vehicles_lane, vehicles_pos,
                                                              vehicle_pos_pygame, vehicles_vel):
                d = get_distance([x_ego, y_ego], [v_pos[0], v_pos[1]])
                
                if d < RADAR_RENGE:

                    if  (DIRECTION == 1 and 4 <= int(vLane) <= 6) or (DIRECTION == -1 and 1 <= int(vLane) <= 3):
                        pygame.draw.rect(screen, RED, v_pos_pygame)

                        # extract the adjacent vehicles positions
                        neighboring_vehicles_pos.append(v_pos)
                        neighboring_vehicles_pos_pygame.append(v_pos_pygame)
                        target_vehicle_info.append((v_pos[0],v_pos[1],v_pos[2],v_pos[3],v_vel[0],v_vel[1]))

                        # Center of rectangle for x, y positions
                        X_pos = v_pos[0] + v_pos[2] / 2
                        Y_pos = v_pos[1] + v_pos[3] / 2

                        # Sending vehicles info to prolog
                        facts = f"vehicle(v{vId}, {vLane}, {X_pos:.4f}, {Y_pos:.4f}, {v_pos[2]:.4f}, {v_pos[3]:.4f}, {v_vel[0]:.4f}, {v_vel[1]:.4f}).\n"
                        f.write(facts)

                    else:
                        pygame.draw.rect(screen, WHITE, v_pos_pygame)

                else:
                        pygame.draw.rect(screen, WHITE, v_pos_pygame)

            # Sending Ego info
            EgoLane = get_lane(agent.y, LANES_Y)

            # factual information about ego
            EgoFact = f"vehicle(ego, {EgoLane}, {agent.x:.4f}, {agent.y:.4f}, {agent.width:.4f}, {agent.height:.4f}, {agent.velocity_x:.4f}, {agent.velocity_y:.4f}).\n"
            f.write(EgoFact)

            N_lc = agent.get_lane_change_number()
            f.write(f"previous_lane_changes({N_lc}).")
            f.close()

        agent.target_vehicle_info = target_vehicle_info

        # reconsult the prolog file to load clauses for finding safe actions
        # you should add 'reconsult' command like 'consult' in pyswip file -> find prolog.py in pyswip installed directory
        prolog.reconsult('prolog/symbolic_logical_programming.pl')

        L = list(prolog.query('states(States)'))
        states = L[0]['States']

        L = list(prolog.query('velocities(Vs)'))
        velocities = L[0]['Vs']

        agent.lc_state = states + [abs(velocities[0])]
        # agent.velocity_state = [states[0], states[4], velocities[1], velocities[5]]

        # Action = list(prolog.query('safe_actions(Action)'))
        # L = list(prolog.query('possible_actions(Actions)'))
        # safeActions = []
        # for action in L[0]['Actions']:
        #     safeActions.append(str(action))
        # # update the safe action set
        # agent.possible_actions = safeActions

        best_action = list(prolog.query('best_action(Action)'))
        agent.best_action = best_action[0]['Action']
        # print(agent.best_action)
        # print(f">>> Safe action set: {safeActions}.")

        # front_is_free = list(prolog.query('front_is_free'))
        # if front_is_free== [{}]:
        #     agent.front_is_free = True
        # else:
        #     agent.front_is_free = False
        # print(agent.front_is_free)

        Desired_velocity_x = list(prolog.query('longitudinal_velocity(Vx)'))
        # update the desired velocity
        agent.V_x_desired = Desired_velocity_x[0]['Vx']
        # print(f">>> Desired Velocity x: {Desired_velocity_x[0]['Vd_x']}.", end="\n")

        L = list(prolog.query("states(State)"))
        states = []
        for state in L[0]['State']:
            states.append(state)
        # update state
        agent.state = states

        EgoRect, EgoColor = agent.rect, agent.color
        pygame.draw.rect(screen, EgoColor, EgoRect)
        pygame.display.update()
        clock.tick(40)

        # update the autonomous vehicle's state
        done = agent.update(timestep, symbolic=SIL, train=TRAIN, 
                            target_vehicles_list=neighboring_vehicles_pos, max_distance=max_distance)

        # end the episode if done 
        if done or agent.epoch == N_EPOCHS:

            if TRAIN:
                frame = 0
            else:
                pass

            # if episode == 0 or agent.n_hits == 0:

            # Total steps
            Steps.append(steps)

            # Number of hits
            n_hits.append(agent.n_hits)
            n_lane_changes.append(int(agent.n_lane_changes))
            traveled_distances.append(agent.traveled_distance)

            # create dataframe to save rewards in excel
            df = pd.DataFrame()

            # specify column name to each list
            df['steps'] = Steps
            df['n_hits'] = n_hits
            df['n_lc'] = n_lane_changes
            df['distance'] = traveled_distances

            # Print important data
            print(f'Episode:{episode + 1}/{N_EPISODES}| LaneChange:{int(agent.n_lane_changes)}|' +
                    f'Hits:{agent.n_hits}| Traveled_distance: {agent.traveled_distance:.4f}| Epoch: {agent.epoch}',
                    end="\n\n")

            # Save dataframe to excel
            if int((episode + 1) % SAVE_DATA_STEPS) == 0:
                df.to_excel(OUTPUT_EXCEL_FILE)

            episode += 1
            steps = 0

            # save weights periodically
            # agent.dq_agent.net.save_weights(WEIGHT_FILE)

            # save memory
            # agent.dq_agent.memory.save_memory(MEMORY_FILE)

            # else:
            #     print("Loading weights and memory ... \n")
            #     agent.dq_agent.net.load_weights(WEIGHT_FILE)
            #     # agent.dq_agent.target_net.load_weights(best_weights_file)
            #     agent.dq_agent.memory.load_memory(MEMORY_FILE)

            # make the summation of each reward subfunction zero
            agent.reset_episode()

        frame += 1
        # restart
        if frame >= len(extractor.frame_list):
            frame = 0

    # save weights at the end of the training
    # agent.dq_agent.net.save_weights("weights/weights_" + str(episode) + '_' + str(timestr))