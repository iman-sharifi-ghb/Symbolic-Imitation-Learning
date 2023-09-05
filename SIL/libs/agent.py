# =====================================================================
# Date  : 1 Aug 2023
# Title : Autonomous Vehicle agent
# Creator: Iman Sharifi
# =====================================================================

import pygame
from libs.IL import ILAgent
from libs.utils import get_lane, distance, get_most_recent_file
import random
import torch

distance = distance()

# lane details
# y values of lanes (do not change)
LANES_Y = [10, 25, 36, 48, 81, 93, 108, 120]
ROAD_LENGTH = 1366.0  # image pixel
REAL_ROAD_LENGTH = 420.0  # meter
TIME_STEP = 1.0 / 25.0
LANE_WIDTH = 10
semi_lane_width = LANE_WIDTH / 2

LANE_CHANGE_STEPS = 8  # time steps required for state transition (lane change)
MAX_SPEED = 120
MAX_LANE = 6
RADAR_RANGE = 70
LANE_CHANGES_HISTORY_LENGTH = 200

# reward details
CAR_HIT_REWARD = -100  # if hit by a car
END_OF_LANE_REWARD = 100  # for reaching to the end of the lane
REWARD_LANE_CHANGE = -1  # reward for unnecessary lane change
REWARD_OUT_OF_LEGAL_LANES = -100  # reward for getting out of the highway lanes
REWARD_TIME_WASTE = 1  # reward for wasting time
REWARD_FRONT_FREE = 3
LANE_CHANGE_STEPS = 8  # time steps required for state transition (lane change)

# =====================================================================
# autonomous vehicle's class
class AgentCar:
    """
    STATES = ['north_car_distance', 'northeast_car_distance', 'east_car_distance', 'southeast_car_distance',
              'south_car_distance', 'southwest_car_distance', 'west_car_distance', 'northwest_car_distance',
              'autonomous_vehicle_lane', 'autonomous_vehicle_velocity']

    Note: each of the substates is normolized between 0 and 1. The distances are divided by the radar range.          

    ACTIONS = ["lane_keeping", "left_lane_change", "right_lane_change"] ~ [0, 1, 2]
    """
    def __init__(self, width=15, height=8, direction=1, symbolic=True):

        self.dt = TIME_STEP
        self.width = width  # width of the car
        self.height = height  # height
        self.direction = direction  # 1 is left-to-right, -1 for right-to-left

        # initialize the DQN agent
        if not symbolic:
            self.lc_agent = ILAgent(state_size=9, n_actions=3, fc1_units=128, fc2_units=128, weight_file_path='')
            self.velocity_agent = ILAgent(state_size=4, n_actions=1, fc1_units=128, fc2_units=128, weight_file_path='')

            self.lc_state = []
            self.velocity_state = []
        
        self.reset_episode()
        self.rect = pygame.Rect(self.x - int(width / 2), self.y - int(height / 2), self.width, self.height)
        self.color = (0, 255, 0)

        # Safe action derived from Prolog
        self.safe_action = 0
        self.possible_actions = []

    def reset_episode(self):
        self.epoch = 0
        self.reset_epoch()

        # get the initial y value
        if self.direction == 1:
            self.lane = random.randint(4, 6)
            self.y = LANES_Y[self.lane] - semi_lane_width
        else:
            self.lane = random.randint(1, 3)
            self.y = LANES_Y[self.lane] - semi_lane_width            

        # reset states
        self.velocity_x = self.initial_speed()
        self.V_x_desired = self.initial_speed()
        self.velocity_y = 0

        self.score = 0  # score of this episode
        self.reward_slp = 0
        self.time = 0  # time of this episode
        self.n_lane_changes = 0  # number of lane changes
        self.n_hits = 0  # number of collisions
        self.n_outs = 0 # number of out of highway

        # the agent is allowed to choose an new action
        self.take_new_action = True

        # state = [north,northeast,east,southeast,south,southwest,west,northwest,lane/maxLane,velocity/maxVelocity]
        # We extract the state from Prolog
        # if there is no car in a location, the corresponding number will be -1. otherwise it is distance/radar_range.
        self.state = [1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.8]

        # state and action of  the previous frame
        self.previous_state = [1, 1, 1, 1, 1, 1, 1, 1, self.lane/MAX_LANE, self.velocity_x/MAX_SPEED]

        # Action delay parameters
        self.previous_action = 0
        self.previous_actions = ['lane_keeping']*LANE_CHANGES_HISTORY_LENGTH
        self.action_repeat = 0
        self.previous_y_desired = self.y
        self.best_action = 'lane_keeping'

        self.traveled_distance = 0

        # target vehicles information
        self.target_vehicle_info = []

        # if there is a vehicle in front of the AV
        self.front_is_free = True

    def reset_epoch(self):
        # get the initial x value
        if self.direction == 1:
            self.x = self.width
        else:
            self.x = ROAD_LENGTH - self.width

        # reset integrator error for PID control of y
        self.e_i_y = 0
        self.e_i_V = 0

    def initial_speed(self):
        if self.lane == 4 or self.lane == 3:
            return 100

        elif self.lane == 5 or self.lane == 2:
            return 90

        elif self.lane == 6 or self.lane == 1:
            return 80

        else:
            return 70

    def reset_lane(self, target_vehicles_list):
        # get the initial y value
        self.lane = self.init_lane(target_vehicles_list)
        self.y = LANES_Y[self.lane] - semi_lane_width

    def init_lane(self, target_vehicles_list):

        if self.direction == 1:
            lane = random.randint(4, 6)
            self.y = LANES_Y[lane] - semi_lane_width

            i = 0
            while i < 10:
                if self.overlap(target_vehicles_list) == 1:
                    lane = random.randint(4, 6)
                else:
                    break
                i += 1

        else:
            lane = random.randint(1, 3)
            self.y = LANES_Y[lane] - semi_lane_width

            i = 0
            while i < 10:
                if self.overlap(target_vehicles_list) == 1:
                    lane = random.randint(1, 3)
                else:
                    break
                i += 1

        return lane

    def overlap(self, target_vehicles_list):
        W, H = self.width, self.height
        X = self.x - int(W / 2)
        Y = self.y - int(H / 2)
        ego_rect = pygame.Rect(X, Y, W, H)

        temp = 0
        for vehicle_rect in target_vehicles_list:
            intersection_rect = ego_rect.clip(vehicle_rect)

            if intersection_rect.w > 0 and intersection_rect.h > 0:
                temp = 1
                break

        return temp
    
    def overlap_(self, x, y, target_vehicles_list):
        W, H = self.width, self.height
        X = x - int(W / 2)
        Y = y - int(H / 2)
        ego_rect = pygame.Rect(X, Y, W, H)

        temp = 0
        for vehicle_rect in target_vehicles_list:
            intersection_rect = ego_rect.clip(vehicle_rect)

            if intersection_rect.w > 0 and intersection_rect.h > 0:
                temp = 1
                break

        return temp

    def get_y_desired(self, lane, action):

        if lane != 0:
            if self.direction == 1:

                if action == 'right_lane_change':
                    y_desired = LANES_Y[lane + 1] - semi_lane_width

                elif action == 'left_lane_change':
                    y_desired = LANES_Y[lane - 1] - semi_lane_width

                else:
                    y_desired = LANES_Y[lane] - semi_lane_width

            else:

                if action == 'right_lane_change':
                    y_desired = LANES_Y[lane - 1] - semi_lane_width

                elif action == 'left_lane_change':
                    y_desired = LANES_Y[lane + 1] - semi_lane_width

                else:
                    y_desired = LANES_Y[lane] - semi_lane_width

        else:
            y_desired = self.y
        
        self.previous_y_desired = y_desired

        return y_desired
    # =====================================================================
    # execute action (y control)
    def perform_action(self, action, state_transition):
        if state_transition:
            y_desired = self.previous_y_desired

        else:
            y_desired = self.get_y_desired(self.lane, action)

        self.velocity_y = self.y_PI_Control(y_desired)

    # =====================================================================
    # convert numeric actions [0, 1, 2] to symbolic actions {LK, LLC, RLC}
    def translate_action(self, action):

        if action == 0:
            return 'lane_keeping'
        
        elif action == 1:
            return 'left_lane_change'
        
        elif action == 2:
            return 'right_lane_change'
        
        elif action == 'lane_keeping':
            return 0
        
        elif action == 'left_lane_change':
            return 1
        
        elif action == 'right_lane_change':
            return 2
        
        else:
            print("Action Error: Action is invalid!\n")
            return 'lane_keeping'

    # =====================================================================
    # adjuct the lateral acceleration (a_y) to control the lateral position (y) when changing the lane 
    def y_PI_Control(self, y_desired):

        Kp, Ki = 4, 2
        e_p = y_desired - self.y
        self.e_i_y += e_p * self.dt
        u = Kp * e_p + Ki * self.e_i_y

        return u

    # =====================================================================
    # adjust the longitudinal acceleration (a_x) to control the longitudinal velocity
    def Vx_PI_Control(self, Vx_desired):

        Kp, Ki = 3, 1
        e_p = Vx_desired - self.velocity_x
        self.e_i_V += e_p * self.dt
        u = Kp * e_p + Ki * self.e_i_V

        return u

    # =====================================================================
    # reward functions
    def get_lane_change_reward(self, action):

        if action == 'left_lane_change' or action == 'right_lane_change':
            if self.front_is_free:
                return REWARD_LANE_CHANGE
            else:
                return -REWARD_LANE_CHANGE
        else:
            return 0

    def get_collision_reward(self, target_vehicles_list):

        reward_collision = 0
        done = False
        W, H = self.width, self.height
        X = self.x - int(W / 2)
        Y = self.y - int(H / 2)
        ego_rect = pygame.Rect(X, Y, W, H)

        if (self.direction ==1 and self.x > 200) or (self.direction == -1 and self.x < (ROAD_LENGTH-200)):

            for vehicle_rect in target_vehicles_list:
                intersection_rect = ego_rect.clip(vehicle_rect)

                if intersection_rect.w > 0 and intersection_rect.h > 0:
                    self.n_hits += 1
                    reward_collision = CAR_HIT_REWARD * (1-0.8*self.x/ROAD_LENGTH)
                    done = True

        return reward_collision, done

    def get_end_of_episode_reward(self, max_distance):
        # End of the lane
        done = False

        if self.traveled_distance >= (max_distance - self.width):
            reward_end = END_OF_LANE_REWARD
            done = True

        else:
            reward_end = 0
            
        return reward_end, done

    def get_offroad_reward(self):
        done = False
        reward_out = 0

        if self.direction == 1:
            Border1, Border2 = 65, LANES_Y[6]

            if Border1 <= self.y <= Border2:
                reward_out = 0

            else:
                done = True
                reward_out = REWARD_OUT_OF_LEGAL_LANES*(1-0.8*self.x/ROAD_LENGTH)

        else:
            Border1, Border2 = LANES_Y[0], LANES_Y[3]
            if Border1 <= self.y <= Border2:
                reward_out = 0

            else:
                done = True
                reward_out = REWARD_OUT_OF_LEGAL_LANES

        return reward_out, done

    def get_lane_reward(self):

        if self.lane == 4 or self.lane == 3:
            lane_reward = 10

        elif self.lane == 5 or self.lane == 2:
            lane_reward = 15

        elif self.lane == 6 or self.lane == 1:
            lane_reward = 0

        else:
            lane_reward = 0

        return lane_reward

    def get_velocity_reward(self):
        if self.velocity_x > 110:
            reward_velocity = 0.1 * (self.velocity_x - 110)
        else:
            reward_velocity = 0
        return reward_velocity

    def get_distance_reward(self):
        return 0.01 * self.x

    def state_action_reward(self, action):
        weights_states = [1, 1, 1, 1, 1, 1, 1, 1, 0, 8]
        weight_action = 1
        desired_state = [1, 1, 1, 1, 1, 1, 1, 1, 5 / 6, 1]
        state_error = [i - j for i, j in zip(desired_state, self.state)]

        desired_action = 'lane_keeping'

        if action == desired_action:
            error_action = 0

        else:
            error_action = 1

        state_error_square = [i * j ** 2 for i, j in zip(weights_states, state_error)]
        R = -sum(state_error_square) - weight_action * error_action

        return R

    def get_front_free_reward(self):
        if self.state[0] == 1:
            return REWARD_FRONT_FREE
        else:
            return 0

    def get_traveled_distance(self):
        if self.direction == 1:
            traveled_distance = self.epoch*ROAD_LENGTH + self.x 
        else:
            traveled_distance = self.epoch*ROAD_LENGTH + ROAD_LENGTH - self.x 

        return traveled_distance

    def end_of_epoch(self):

        done = False

        if self.direction == 1:
            if self.x >= (ROAD_LENGTH - self.width):
                done = True
        else:
            if self.x <= self.width:
                done = True
        
        return done
    
    def update_lane(self):
        self.lane = get_lane(self.y, LANES_Y)

    def predict_next_state(self, x, v_x, lane, action):
        x_next = x + v_x * TIME_STEP
        y_next = self.get_y_desired(lane, action)

        return x_next, y_next
    
    def Degree_of_Safety(self, action, target_vehicles):

        x_next_ego, y_next_ego = self.predict_next_state(self.x, self.velocity_x, self.lane, action)
        ego_next_state = (x_next_ego, y_next_ego, self.width, self.height)

        target_vehicles_next_states = []
        target_vehicles_next_states_pygame = []

        if action == "lane_keeping":
            temp = 1
        else:
            temp = LANE_CHANGE_STEPS

        for vehicle in target_vehicles:
            x, y, w, h, vx, vy = vehicle[0], vehicle[1], vehicle[2], vehicle[3], vehicle[4], vehicle[5]
            x_next, y_next = x, y
            for i in range(temp):
                x_next += vx*TIME_STEP
                y_next += vy*TIME_STEP
            target_vehicles_next_states.append((x_next,y_next, w, h))
            target_vehicles_next_states_pygame.append(pygame.Rect(x_next,y_next, w, h))

        # offroad safety degree
        _, offroad = self.get_offroad_reward()
        if offroad:
            offroad_DoS = 0
        else:
            offroad_DoS = 1

        # collision safety degree
        collision = self.overlap_(x_next_ego, y_next_ego, target_vehicles_next_states_pygame)
        if collision:
            collision_DoS = 0
        else:
            collision_DoS = 1

        # distance safety degree
        distance_DoS = 1
        R = RADAR_RANGE

        if len(target_vehicles_next_states) > 0:
            for vehicle in target_vehicles_next_states:
                d = distance.absolute_distance(p1=ego_next_state, p2=vehicle)
                distance_DoS *= d/R

        # lane change safety degree
        lane_change_DoS = 1
        if action != "lane_keeping":
            for act in self.previous_actions[-LANE_CHANGES_HISTORY_LENGTH:-1]:
                if act != "lane_keeping":
                    lane_change_DoS *= 0.5

        # velocity degree of safety
        next_lane_DoS = 1
        next_lane_ego = get_lane(y_next_ego, LANES_Y)
        if next_lane_ego == 4 or next_lane_ego == 3:
            next_lane_DoS *= 0.8
        elif next_lane_ego == 6 or next_lane_ego == 1:
            next_lane_DoS *= 0.3
        elif next_lane_ego == 5 or next_lane_ego == 2:
            next_lane_DoS *= 0.95
        else:
            next_lane_DoS *= 0

        # front_is_busy safety degree
        front_DoS = 1
        if not(self.front_is_free) and action == 'lane_keeping':
            front_DoS *= 0.1

        return offroad_DoS * collision_DoS * distance_DoS * lane_change_DoS * next_lane_DoS * front_DoS

    def get_best_action(self, safe_actions, target_vehicles):
        action_DoFs = []
        for action in safe_actions:
            DoS = self.Degree_of_Safety(action, target_vehicles)
            action_DoFs.append(DoS)        
        # print(action_DoFs, safe_actions[action_DoFs.index(max(action_DoFs))])

        return safe_actions[action_DoFs.index(max(action_DoFs))]

    def get_safe_action_indexes(self, actions, safe):
        if safe:
            # assign [0, 1, 2] to [lane_keeping, left_lane_change, right_lane_change] in the safe action set
            indexes = []
            for action in actions:
                if action == 'lane_keeping':
                    indexes.append(0)
                elif action == 'left_lane_change':
                    indexes.append(1)
                elif action == 'right_lane_change':
                    indexes.append(2)
                else:
                    indexes.append(0)
        else:
            # DQN can choose both safe and unsafe actions
            indexes = [0, 1, 2]

        return indexes

    def get_lane_change_number(self):
        N = 0
        for act in self.previous_actions[-LANE_CHANGES_HISTORY_LENGTH:-1]:
            if act != "lane_keeping":
                N += 1
        return N

    # =====================================================================
    # update the autonomous vehicle agent state in each frame by taking new actions
    # return               = done, score
    # dt                   = frame time
    # target_vehicles_list = list of the detected vehicles by the radar installed on the Autonomous Vehicle
    # train                = enable or disable learning
    def update(self, dt, symbolic=True, train=False, target_vehicles_list=[], max_distance=ROAD_LENGTH):
        done, done1, done2, done3, done4 = False, False, False, False, False
        state_transition = False

        # print(f"Ego position: X = {self.x:.4f}, Y = {self.y:.4f}, Lane = {self.lane}.")
        # print(f"Ego Velocity: Vx = {self.velocity_x:.4f}, Vy = {self.velocity_y:.4f}, Lane = {self.lane}.")

        # (Safe) Action =====================================
        if self.take_new_action:
            # Safe action set
            # best_action = self.get_best_action(self.possible_actions, self.target_vehicle_info)
            if symbolic:
                best_action = self.best_action
            else:
                state_tensor = torch.tensor([self.lc_state], dtype=torch.float32)
                il_action = self.lc_agent.forward(state_tensor, softmax=True)
                il_action = il_action.tolist()[0]
                # print(il_action)
                # if abs(max(il_action)-min(il_action)) > 0.2:
                il_action_index = il_action.index(max(il_action))
                # else:
                #     il_action_index = 0
                best_action = self.translate_action(il_action_index)

            action = self.translate_action(best_action)

            # get distances to surrounding cars
            state = self.state
            # safe_actions_indexes = self.get_safe_action_indexes(self.possible_actions, safe)
            # best_action_index = self.get_safe_action_indexes([best_action],safe)
            # action = self.translate_action(best_action)
            learn = train

            self.previous_action = action  # Save previous action
            self.previous_state = state   # Save previous state

        else:
            action = self.previous_action
            state = self.previous_state

        # Execute a time delay to change the lane completely
        if action != 0 and self.action_repeat < LANE_CHANGE_STEPS:
            if self.action_repeat == 0:
                self.n_lane_changes += 1
                self.previous_y_desired = self.get_y_desired(self.lane, self.translate_action(action))

            self.take_new_action = False
            state = self.previous_state
            action = self.previous_action
            learn = False  
            state_transition = True

            self.action_repeat += 1
            # Reset action repeat
            if self.action_repeat == LANE_CHANGE_STEPS:
                learn = train
                self.take_new_action = True
                self.action_repeat = 0

        # print(learn)
        # Map [0, 1, 2] to [lane_keeping, left_lane_change, right_lane_change]
        symbolic_action = self.translate_action(action)
        self.previous_actions.append(symbolic_action)

        _, done1 = self.get_collision_reward(target_vehicles_list)
        _, done2 = self.get_offroad_reward()
        _, done3 = self.get_end_of_episode_reward(max_distance)
        done = done1 or done2 or done3

        # Learning Process ======================================
        # if learn is enabled
        # if learn:
        #     pass
            # Update the net weights ============================
            # self.dq_agent.learn(episode)

        if done2:
            self.n_outs += 1

        done4 = self.end_of_epoch()
        if done4:
            self.reset_epoch()
            self.epoch += 1

        # if terminal state met
        if not done:
            # Velocity Control =================================
            # if symbolic:
            ax = self.Vx_PI_Control(self.V_x_desired)
            self.velocity_x += ax * dt
            # else:
            #     v_state_tensor = torch.tensor(self.velocity_state, dtype=torch.float32)
            #     il_velocity = self.velocity_agent.forward(v_state_tensor, softmax=False)
            #     il_velocity = il_velocity.tolist()[0]*MAX_SPEED
            #     # print(abs(il_velocity))
            #     self.velocity_x = abs(il_velocity)
            
            # Execute action ==================================
            # self.perform_action(self.safe_action)
            state_transition = not(learn)
            self.perform_action(symbolic_action, state_transition)

            # Update longitudinal and lateral positions of Ego Vehicle and the assigned rectangle
            self.x += self.velocity_x * dt * self.direction
            self.y += self.velocity_y * dt * self.direction
            self.rect.center = (self.x, self.y)

        # time
        self.time += dt

        # update lane in each step
        self.update_lane()

        # traveled distance
        self.traveled_distance = self.get_traveled_distance()

        return done
