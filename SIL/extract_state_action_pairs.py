# =====================================================================
# Date  : 1 Aug 2023
# Title : extract state-action pairs
# Creator : Iman Sharifi
# =====================================================================

from decimal import Decimal
import pandas as pd
import csv
import pygame
import sys
import os
from pygame.locals import *
# from agent import AgentCar
import matplotlib.pyplot as plt
import math
import numpy as np
import time
from libs.utils import get_distance, get_lane, write_facts, prolog_query
# from libs.utils import translate_outputs_to_facts, write_setting, write_bk, write_example
# from libs.utils import create_frame_list, get_car_list, lane_change_detection
from libs.utils import data_extractor
from pyswip import Prolog
import warnings

warnings.filterwarnings('ignore')
prolog = Prolog()

# Adjustable parameters ============================================
DATASET_DIRECTORY = '../dataset/'
OUTPUT_EXCEL_FILE = 'excel/state_action_pair.xlsx'

# ==================================================================
RADAR_RANGE = 150 # PIXEL
MAX_SPEED = 120
LANES_Y = [10, 25, 36, 48, 81, 93, 108, 120]

# =====================================================================
# main
if __name__ == '__main__':

    # the dimentional parameters of the real-world highway and the scaled highway in pygame
    road_length, road_width, X, Y = 420, 36.12, 1366, 118
    timestep = 1 / 25

    # initialize data-extractor 
    extractor = data_extractor(directory=DATASET_DIRECTORY)

    # Pygame Settings ===============================================, 
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)

    # pygame.init()  # initialize pygame
    # clock = pygame.time.Clock()
    # screen = pygame.display.set_mode((X, Y))

    # # Load the background image here. Make sure the file exists!
    # bg = pygame.image.load(DATASET_DIRECTORY + "highway.jpg")
    # bg = pygame.transform.scale(bg, (X, Y))
    # pygame.mouse.set_visible(1)
    # pygame.display.set_caption('Highway')
    # font = pygame.font.SysFont('Arial', 10)

    ego_exists = False
    flag = 0

    # get general info of highD dataset
    all_vehicle_ids, initialFrames, FinalFrames = extractor.get_general_info()

    # lane change info
    # lane_change_info = lane_change_detection(DATASET_DIRECTORY)
    lane_change_info = extractor.lane_change_detection()
    detected_vehicles = [info[0] for info in lane_change_info]
    detected_frames = [info[1] for info in lane_change_info]
    detected_actions = [info[3] for info in lane_change_info]

    # frame counter
    frame = 0

    # number of scenarios for lane changes
    N = 0
    States, Actions = [], []

    for Vehicle_id, initFrame, finFrame in zip(all_vehicle_ids, initialFrames, FinalFrames):
        ego_id = int(Vehicle_id)
        if ego_id in detected_vehicles:
            detected_frame = detected_frames[detected_vehicles.index(ego_id)]
            lane_change_frames = [i for i in range(detected_frame-20, detected_frame+50)]
        
        # the main training loop =======================================
        for frame in range(initFrame+1, finFrame, 4):
            
            if ego_id in detected_vehicles and frame in lane_change_frames:
                # detect taken action
                action = detected_actions[detected_vehicles.index(ego_id)]
                if action == 'right_lane_change':
                    act = 2
                else:
                    act = 1
            else:
                act = 0

            # get each vehicle's info
            x_ego = extractor.get_position(ego_id, frame)[0]
            y_ego = extractor.get_position(ego_id, frame)[1]
            w_ego = extractor.get_position(ego_id, frame)[2]
            h_ego = extractor.get_position(ego_id, frame)[3]
            X_ego = x_ego + w_ego / 2
            Y_ego = y_ego + h_ego / 2
            egoLane = get_lane(y_ego, LANES_Y)
            vx_ego = extractor.get_velocity(ego_id, frame)[0]
            vy_ego = extractor.get_velocity(ego_id, frame)[1]
            ax_ego = extractor.get_acceleration(ego_id, frame)[0]

            # previous longitudinal velocity
            vx_ego_previous = extractor.get_velocity(ego_id, frame-1)[0]
            ego_pos_pygame = pygame.Rect((x_ego,y_ego,w_ego,h_ego))

            # prepare facts (ego and target vehicles) for further processes in prolog
            facts = []
            ego_fact = f"vehicle(ego, {egoLane}, {X_ego:.4f}, {Y_ego:.4f}, {w_ego:.4f}, {h_ego:.4f}, {vx_ego:.4f}, {vy_ego:.4f})."
            facts.append(ego_fact)

            # get vehicles' info in a single frame
            vehicles_id = extractor.get_ids(frame)
            vehicles_lane = extractor.get_lanes(frame)
            vehicles_direction = extractor.get_directions(frame)
            vehicles_pos = extractor.get_positions(frame)
            vehicles_vel = extractor.get_velocities(frame)
            vehicles_acc = extractor.get_accelerations(frame)
            action_validities = extractor.get_action_validities(frame)
            busy_sections = extractor.get_busy_sections(frame)

            vehicle_pos_pygame = [pygame.Rect(rect) for rect in vehicles_pos]

            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         sys.exit()
            # screen.blit(bg, (0, 0))

            # finding target vehicles
            zipped_info = zip(vehicles_id, vehicles_lane, vehicles_direction, vehicles_pos, vehicle_pos_pygame, vehicles_vel)
            for vId, vLane, v_dir, v_pos, v_pos_pygame, v_vel in zipped_info:
                vId, vLane, v_dir = int(vId), int(vLane), int(v_dir)

                if vId != ego_id:
                    d = get_distance([x_ego,y_ego], [v_pos[0],v_pos[1]])
                    if 0 < d < RADAR_RANGE:
                        if (4 <= vLane <= 6 and 4 <= egoLane <= 6) or (1 <= vLane <= 3 and 1 <= egoLane <= 3):
                            # pygame.draw.rect(screen, RED, v_pos_pygame)

                            # Center of rectangle for x, y positions
                            X_pos = v_pos[0] + v_pos[2] / 2
                            Y_pos = v_pos[1] + v_pos[3] / 2

                            # Sending vehicles info to prolog
                            fact = f"vehicle(v{vId}, {vLane}, {X_pos:.4f}, {Y_pos:.4f}, {v_pos[2]:.4f}, {v_pos[3]:.4f}, {v_vel[0]:.4f}, {v_vel[1]:.4f})."
                            facts.append(fact)
                #     else:
                #         pygame.draw.rect(screen, WHITE, v_pos_pygame)
                # else:
                #     pygame.draw.rect(screen, GREEN, ego_pos_pygame)

            # write facts into a prolog file
            write_facts(file_name='prolog/vehicles_info.pl', facts=facts)
            
            highway_rule_file='prolog/symbolic_logical_programming.pl'
            prolog.reconsult(highway_rule_file)
            L = list(prolog.query('states(States)'))
            states = L[0]['States']
            L = list(prolog.query('velocities(Vs)'))
            velocities = L[0]['Vs']

            States.append(states + velocities + [ax_ego])
            Actions.append([act, vx_ego/MAX_SPEED])

            # pygame.display.update()
            # clock.tick(40)

        print(ego_id)
        df = pd.DataFrame()
        df['state'] = States
        df['action'] = Actions
        if ego_id % 200 == 0:
            df.to_excel(OUTPUT_EXCEL_FILE)

    df.to_excel(OUTPUT_EXCEL_FILE)
        