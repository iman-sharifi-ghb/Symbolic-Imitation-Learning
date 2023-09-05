import os
import math
from pygame.locals import *
import csv
import pandas as pd
from decimal import Decimal
from pyswip import Prolog

prolog = Prolog()

# the dimentional parameters of the real-world highway and the scaled highway in pygame
ROAD_LENGTH, ROAD_WIDTH, X_PIXEL, Y_PIXEL = 420, 36.12, 1366, 118
LANES_Y = [10, 25, 36, 48, 81, 93, 108, 120]

# =====================================================================
class distance:
    def __init__(self):
        pass

    def eucleadian_distance(self, p1=(0,0), p2=(0,0)):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def longitudinal_distance(self, p1=(0,0,0,0), p2=(0,0,0,0)):
        d = abs(p1[0]-p2[0]) - (p1[2]+p2[2])
        if d > 0:
            return d
        else:
            return 0
        
    def lateral_distance(self, p1=(0,0,0,0), p2=(0,0,0,0)):
        d = abs(p1[1]-p2[1]) - (p1[3]+p2[3])
        if d > 0:
            return d
        else:
            return 0
        
    def absolute_distance(self, p1=(0,0,0,0), p2=(0,0,0,0)):
        x_distance = self.longitudinal_distance(p1=p1, p2=p2)
        y_distance = self.lateral_distance(p1=p1, p2=p2)
        return math.sqrt(x_distance**2 + y_distance**2)

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)    

def get_lane(y, LANES_Y):

    if LANES_Y[0] <= y <= LANES_Y[1]:
        return 1
    elif LANES_Y[1]< y <=LANES_Y[2]:
        return 2
    elif LANES_Y[2]< y <=LANES_Y[3]:
        return 3
    elif       65  < y <=LANES_Y[4]:
        return 4
    elif LANES_Y[4]< y <=LANES_Y[5]:
        return 5
    elif LANES_Y[5]< y <=LANES_Y[6]:
        return 6
    else:
        return 0

def write_facts(file_name='', facts=[]):
    with open(file_name, 'w') as f:
        for fact in facts:
            f.write(fact+'\n')
    f.close()

def prolog_query(file_name='', queries=[], variable='A'):
    prolog.reconsult(file_name)
    output = []
    for query in queries:
        query = f"{query}({variable})"
        L = list(prolog.query(query))
        output.append(L[0][variable])
    return output

def translate_outputs_to_facts(outputs):
    sections = ['front', 'front_right', 'right', 'back_right',
                'back', 'back_left', 'left', 'front_left'] 
    sections1 = outputs[:8]
    sections2 = []
    for sec, sec1 in zip(sections, sections1):
        if sec1 == 1:
            temp = 'busy'
        else:
            temp = 'free'
        sections2.append(f'{sec}_is_{temp}')

    validity = ['right', 'left']
    validity1 = outputs[8:10]
    validity2 = []
    for sec, sec1 in zip(validity, validity1):
        if sec1 == 1:
            temp = 'valid'
        else:
            temp = 'invalid'
        validity2.append(f'{sec}_is_{temp}')

    velocity = sections
    velocity1 = outputs[10:]
    velocity2 = []
    for sec, sec1 in zip(velocity, velocity1):
        if sec1 == 1:
            temp = 'bigger'
            velocity2.append(f'{sec}_velocity_is_{temp}')
        elif sec1 == 0:
            temp = 'equal'
            velocity2.append(f'{sec}_velocity_is_{temp}')
        elif sec1 == -1:
            temp = 'lower'
            velocity2.append(f'{sec}_velocity_is_{temp}')
        else:
            pass
        
    return sections2 + validity2 + velocity2

def settings(head=''):
    head_pred = [f'head_pred({head},1).']

    sections = ['front', 'front_right', 'right', 'back_right',
                'back', 'back_left', 'left', 'front_left']
    body_preds = []
    for section in sections:
        body_preds.append(f'body_pred({section}_is_busy,1).')
        body_preds.append(f'body_pred({section}_is_free,1).')
        body_preds.append(f'body_pred({section}_velocity_is_bigger,1).')
        body_preds.append(f'body_pred({section}_velocity_is_equal,1).')
        body_preds.append(f'body_pred({section}_velocity_is_lower,1).')

    sides = ['right', 'left']
    for side in sides:
        body_preds.append(f'body_pred({side}_is_valid,1).')
        body_preds.append(f'body_pred({side}_is_invalid,1).')

    return head_pred + body_preds
        
def write_setting(file_name='bias.pl', head=''):
    preds = settings(head=head)
    with open(file_name, 'w') as f:
        for pred in preds:
            f.write(pred+'\n')
    f.close()

def write_bk(file_name='bk.pl', scenario=[], N=0):
    if N < 2:
        overwrite = 'w'
    else:
        overwrite = 'a'

    with open(file_name, overwrite) as f:
        if N == 1:
            f.write(':- style_check(-discontiguous).\n')
        f.write(f'% scenario no. {N}:\n')
        for scene in scenario:
            f.write(f'{scene}(s{N}).\n')
        f.write('\n')
        f.close()

def write_example(file_name='exs.pl', predicate='', N=0):
    if N < 2:
        overwrite = 'w'
    else:
        overwrite = 'a'

    with open(file_name, overwrite) as f:
        if N == 1:
            f.write(':- style_check(-discontiguous).\n')
        f.write(f'neg({predicate}(s{N})).\n')
    f.close()

# =====================================================================
# create the frame list from the csv files
def create_frame_list(DATASET_DIRECTORY):
    recordingMeta = csv.DictReader(open(DATASET_DIRECTORY + "recordingMeta.csv")).__next__()
    tracks = csv.DictReader(open(DATASET_DIRECTORY + "tracks.csv"))

    frame_list = []
    frame_len = int(Decimal(recordingMeta['duration']) * int(recordingMeta['frameRate']))
    for i in range(frame_len):
        frame_list.append([])

    print('Loading frame list ...')
    i = 0
    for row in tracks:
        frame_list[int(row['frame']) - 1].append(row)
        i = i + 1

    return frame_list

# =====================================================================
# create the car list for a frame
def get_car_list(frame_list, frame):

    vehicles_pos, vehicles_vel, vehicles_acc = [], [], []
    vehicles_id, vehicles_lane, vehicles_direction = [], [], []
    busy_sections = []
    action_validity = []

    for row in frame_list[frame]:
        # each vehicle information
        v_id = row['id']
        a = float(row['x']) * X_PIXEL / ROAD_LENGTH
        b = (float(row['y']) + 0.8) * Y_PIXEL / ROAD_WIDTH
        w = float(row['width']) * X_PIXEL / ROAD_LENGTH
        h = float(row['height']) * Y_PIXEL / ROAD_WIDTH

        velocity_x = float(row['xVelocity']) * X_PIXEL / ROAD_LENGTH
        velocity_y = float(row['yVelocity']) * Y_PIXEL / ROAD_WIDTH
        acc_x = float(row['xAcceleration'])* X_PIXEL / ROAD_LENGTH
        acc_y = float(row['yAcceleration'])* Y_PIXEL / ROAD_WIDTH

        lane = get_lane(b, LANES_Y)
        vehicles_id.append(row['id'])
        vehicles_lane.append(lane)
        vehicles_pos.append((a, b, w, h))
        vehicles_vel.append((velocity_x, velocity_y))
        vehicles_acc.append((acc_x, acc_y))

        # directions 
        if 1 <= lane <= 3:
            vehicles_direction.append(-1)
        elif 4 <= lane <= 6:
            vehicles_direction.append(1)
        else:
            vehicles_direction.append(0)

        # action validity
        if lane == 3 or lane == 4:
            left_is_valid = 0
        else:
            left_is_valid = 1

        if lane == 1 or lane == 6:
            right_is_valid = 0
        else:
            right_is_valid = 1
        action_validity.append((right_is_valid, left_is_valid))

        # ===================================================================
        # front vehicle
        fv_id = row['precedingId']
        if fv_id == 0:
            front_is_busy = 0
        else:
            front_is_busy = 1

        # front right vehicle
        frv_id = row['rightPrecedingId']
        if frv_id == 0:
            front_right_is_busy = 0
        else:
            front_right_is_busy = 1

        # right vehicle
        rv_id = row['rightAlongsideId']
        if rv_id == 0:
            right_is_busy = 0
        else:
            right_is_busy = 1
        
        # back right vehicle 
        brv_id = row['rightFollowingId']
        if brv_id == 0:
            back_right_is_busy = 0
        else:
            back_right_is_busy = 1

        # back vehicle
        bv_id = row['followingId']
        if bv_id == 0:
            back_is_busy = 0
        else:
            back_is_busy = 1

        # back left vehicle 
        blv_id = row['leftFollowingId']
        if blv_id == 0:
            back_left_is_busy = 0
        else:
            back_left_is_busy = 1

        # left adjacent vehicle
        lv_id = row['leftAlongsideId']
        if lv_id == 0:
            left_is_busy = 0
        else:
            left_is_busy = 1

        # front left vehicle 
        flv_id = row['leftPrecedingId']
        if flv_id == 0:
            front_left_is_busy = 0
        else:
            front_left_is_busy = 1

        busy_sections.append((front_is_busy,front_right_is_busy,right_is_busy,\
                              back_right_is_busy,back_is_busy,back_left_is_busy,\
                              left_is_busy,front_left_is_busy))

    return vehicles_id, vehicles_lane, vehicles_direction, vehicles_pos, vehicles_vel, vehicles_acc, busy_sections


def lane_change_detection(DATASET_DIRECTORY):
    tracks = pd.read_csv(DATASET_DIRECTORY + "tracks.csv")
    tracksMeta = pd.read_csv(DATASET_DIRECTORY + "tracksMeta.csv")

    # changed-lane vehicles' id
    vehicles_changed_lane_id = tracksMeta[tracksMeta['numLaneChanges']==1]['id'].to_list()

    # driving directions
    vehicles_direction = tracksMeta[tracksMeta['numLaneChanges']==1]['drivingDirection'].to_list()

    lane_change_info = []
    for v_id, v_dir in zip(vehicles_changed_lane_id, vehicles_direction):

        # extract info of a vehicle
        v_tracks = tracks[tracks['id'] == v_id]

        frames = v_tracks['frame'].to_list()
        delay = 40
        # print(v_tracks.head(5))
        lanes_list = v_tracks['laneId'].to_list()
        first_frame_lane = lanes_list[0]
        final_frame_lane = lanes_list[-1]
        L1 = lanes_list[:-1]
        L2 = lanes_list[1:]
        lane_diff = [abs(l1-l2) for l1, l2 in zip(L1, L2)]
        desired_frame = frames[lane_diff.index(1)]-delay

        # absolute maximum of yAcceleration
        max_acc_tracks = v_tracks.loc[v_tracks.abs().idxmax()['yAcceleration']]

        # lane change frame
        v_frame = int(max_acc_tracks['frame'])

        if v_dir == 2: # left-to-right direction
            if final_frame_lane > first_frame_lane: # max_acc_tracks['yAcceleration'] > 0:
                action = 'right_lane_change'
            else:
                action = 'left_lane_change'

        else:   # right-to-left direction
            if final_frame_lane > first_frame_lane: # max_acc_tracks['yAcceleration'] > 0:
                action = 'left_lane_change'
            else:
                action = 'right_lane_change'

        lane_change_info.append((v_id, desired_frame, v_dir, action))
    
    return lane_change_info


class data_extractor():
    def __init__(self, directory=''):
        self.dataset_directory = directory
        self.load_data()
        self.create_frame_list()

    def load_data(self):
        # reading track files
        self.recordingMeta = csv.DictReader(open(self.dataset_directory + "recordingMeta.csv")).__next__()
        self.tracks = csv.DictReader(open(self.dataset_directory + "tracks.csv"))
        self.tracks_df = pd.read_csv(self.dataset_directory + "tracks.csv")
        self.tracksMeta = pd.read_csv(self.dataset_directory + "tracksMeta.csv")

    def create_frame_list(self):
        # create empty frame list
        frame_list = []
        frame_len = int(Decimal(self.recordingMeta['duration']) * int(self.recordingMeta['frameRate']))
        for i in range(frame_len):
            frame_list.append([])

        # filling the blanks of frame list
        i = 0
        for row in self.tracks:
            frame_list[int(row['frame']) - 1].append(row)
            i = i + 1

        self.frame_list = frame_list

    def get_general_info(self):
        all_ids = self.tracksMeta['id']
        initial_frames = self.tracksMeta['initialFrame']
        final_frames = self.tracksMeta['finalFrame']
        return all_ids, initial_frames, final_frames

    def get_ids(self, frame):
        vehicles_id = []
        for row in self.frame_list[frame]:
            vehicles_id.append(row['id'])
        return vehicles_id
    
    def get_lanes(self, frame):
        vehicles_lane = []
        for row in self.frame_list[frame]:
            b = (float(row['y']) + 0.8) * Y_PIXEL / ROAD_WIDTH
            lane = get_lane(b, LANES_Y)
            vehicles_lane.append(lane)
        return vehicles_lane

    def get_positions(self, frame):
        vehicles_pos = []
        for row in self.frame_list[frame]:
            a = float(row['x']) * X_PIXEL / ROAD_LENGTH
            b = (float(row['y']) + 0.8) * Y_PIXEL / ROAD_WIDTH
            w = float(row['width']) * X_PIXEL / ROAD_LENGTH
            h = float(row['height']) * Y_PIXEL / ROAD_WIDTH
            vehicles_pos.append((a, b, w, h)) 
        return vehicles_pos
    
    def get_position(self, vId, frame):
        for row in self.frame_list[frame]:
            if int(row['id']) == int(vId):
                break
        a = float(row['x']) * X_PIXEL / ROAD_LENGTH
        b = (float(row['y']) + 0.8) * Y_PIXEL / ROAD_WIDTH
        w = float(row['width']) * X_PIXEL / ROAD_LENGTH
        h = float(row['height']) * Y_PIXEL / ROAD_WIDTH
        return (a, b, w, h)          
        
    def get_velocities(self, frame):
        vehicles_vel = []
        for row in self.frame_list[frame]:
            velocity_x = float(row['xVelocity']) * X_PIXEL / ROAD_LENGTH
            velocity_y = float(row['yVelocity']) * Y_PIXEL / ROAD_WIDTH
            vehicles_vel.append((velocity_x, velocity_y))
        return vehicles_vel
        
    def get_velocity(self, vId, frame):
        for row in self.frame_list[frame]:
            if int(row['id']) == vId:
                break
        velocity_x = float(row['xVelocity']) * X_PIXEL / ROAD_LENGTH
        velocity_y = float(row['yVelocity']) * Y_PIXEL / ROAD_WIDTH
        return (velocity_x, velocity_y)
        
    def get_accelerations(self, frame):
        vehicles_acc = []
        for row in self.frame_list[frame]:
            acc_x = float(row['xAcceleration'])* X_PIXEL / ROAD_LENGTH
            acc_y = float(row['yAcceleration'])* Y_PIXEL / ROAD_WIDTH
            vehicles_acc.append((acc_x, acc_y))
        return vehicles_acc
    
    def get_acceleration(self, vId, frame):
        for row in self.frame_list[frame]:
            if int(row['id']) == vId:
                break
        acceleration_x = float(row['xAcceleration']) * X_PIXEL / ROAD_LENGTH
        acceleration_y = float(row['yAcceleration']) * Y_PIXEL / ROAD_WIDTH
        return (acceleration_x, acceleration_y)
    
    def get_directions(self, frame):
        vehicles_direction = []
        lanes = self.get_lanes(frame)
        for lane in lanes:
            if 1 <= lane <= 3:  # right-to-left direction
                vehicles_direction.append(-1)
            elif 4 <= lane <= 6:  # left-to-right direction
                vehicles_direction.append(1)
            else:
                vehicles_direction.append(0)
        return vehicles_direction
    
    def get_action_validities(self, frame):
        action_validity = []
        lanes = self.get_lanes(frame)
        for lane in lanes:
            # left_lane_change validity
            if lane == 3 or lane == 4:
                left_is_valid = 0
            else:
                left_is_valid = 1
            
            # right_lane_change validity
            if lane == 1 or lane == 6:
                right_is_valid = 0
            else:
                right_is_valid = 1

            action_validity.append((right_is_valid, left_is_valid))
        return action_validity

    def get_action_validity(self, vId,  frame):
        for row in self.frame_list[frame]:
            if int(row['id']) == int(vId):
                break
        y = (float(row['y'])+0.8)*Y_PIXEL/ROAD_LENGTH
        lane = get_lane(y, LANES_Y)

        # left_lane_change validity
        if lane == 3 or lane == 4:
            left = 'left_is_invalid'
        else:
            left = 'left_is_valid'
        
        # right_lane_change validity
        if lane == 1 or lane == 6:
            right = 'right_is_invalid'
        else:
            right = 'right_is_valid'

        return (right, left)

    def get_busy_sections(self, frame):
        busy_sections = []
        for row in self.frame_list[frame]:

            # front vehicle
            fv_id = row['precedingId']
            if fv_id == 0:
                front_is_busy = 0
            else:
                front_is_busy = 1

            # front right vehicle
            frv_id = row['rightPrecedingId']
            if frv_id == 0:
                front_right_is_busy = 0
            else:
                front_right_is_busy = 1

            # right vehicle
            rv_id = row['rightAlongsideId']
            if rv_id == 0:
                right_is_busy = 0
            else:
                right_is_busy = 1
            
            # back right vehicle 
            brv_id = row['rightFollowingId']
            if brv_id == 0:
                back_right_is_busy = 0
            else:
                back_right_is_busy = 1

            # back vehicle
            bv_id = row['followingId']
            if bv_id == 0:
                back_is_busy = 0
            else:
                back_is_busy = 1

            # back left vehicle 
            blv_id = row['leftFollowingId']
            if blv_id == 0:
                back_left_is_busy = 0
            else:
                back_left_is_busy = 1

            # left adjacent vehicle
            lv_id = row['leftAlongsideId']
            if lv_id == 0:
                left_is_busy = 0
            else:
                left_is_busy = 1

            # front left vehicle 
            flv_id = row['leftPrecedingId']
            if flv_id == 0:
                front_left_is_busy = 0
            else:
                front_left_is_busy = 1

            busy_sections.append((front_is_busy,front_right_is_busy,right_is_busy,\
                                back_right_is_busy,back_is_busy,back_left_is_busy,\
                                left_is_busy,front_left_is_busy))

        return busy_sections    

    def get_busy_section(self, vId, frame):
        for row in self.frame_list[frame]:
            if int(row['id']) == int(vId):
                break

        # front vehicle
        fv_id = row['precedingId']
        if int(fv_id) == 0:
            front = 'front_is_free'
        else:
            front = 'front_is_busy'

        # front right vehicle
        frv_id = row['rightPrecedingId']
        if int(frv_id) == 0:
            front_right = 'front_right_is_free'
        else:
            front_right = 'front_right_is_busy'

        # right vehicle
        rv_id = row['rightAlongsideId']
        if int(rv_id) == 0:
            right = 'right_is_free'
        else:
            right = 'right_is_busy'
        
        # back right vehicle 
        brv_id = row['rightFollowingId']
        if int(brv_id) == 0:
            back_right = 'back_right_is_free'
        else:
            back_right = 'back_right_is_busy'

        # back vehicle
        bv_id = row['followingId']
        if int(bv_id) == 0:
            back = 'back_is_free'
        else:
            back = 'back_is_busy'

        # back left vehicle 
        blv_id = row['leftFollowingId']
        if int(blv_id) == 0:
            back_left = 'back_left_is_free'
        else:
            back_left = 'back_left_is_busy'

        # left adjacent vehicle
        lv_id = row['leftAlongsideId']
        if int(lv_id) == 0:
            left = 'left_is_free'
        else:
            left = 'left_is_busy'

        # front left vehicle 
        flv_id = row['leftPrecedingId']
        if int(flv_id) == 0:
            front_left = 'front_left_is_free'
        else:
            front_left = 'front_left_is_busy'

        busy_sections = (front,front_right,right,back_right,back,back_left,left,front_left)
        return busy_sections 

    def lane_change_detection(self):
        delay = 40

        # changed-lane vehicles' id
        vehicles_changed_lane_id = self.tracksMeta[self.tracksMeta['numLaneChanges']==1]['id'].to_list()

        # driving directions
        vehicles_direction = self.tracksMeta[self.tracksMeta['numLaneChanges']==1]['drivingDirection'].to_list()

        lane_change_info = []
        for v_id, v_dir in zip(vehicles_changed_lane_id, vehicles_direction):

            # extract info of a vehicle
            v_tracks = self.tracks_df[self.tracks_df['id'] == v_id]

            # all frames for a changed-lane vehicle
            frames = v_tracks['frame'].to_list()

            # print(v_tracks.head(5))
            lanes_list = v_tracks['laneId'].to_list()
            first_frame_lane = lanes_list[0]
            final_frame_lane = lanes_list[-1]
            L1 = lanes_list[:-1]
            L2 = lanes_list[1:]
            lane_diff = [abs(l1-l2) for l1, l2 in zip(L1, L2)]
            desired_frame = frames[lane_diff.index(1)]-delay

            # absolute maximum of yAcceleration
            max_acc_tracks = v_tracks.loc[v_tracks.abs().idxmax()['yAcceleration']]

            # lane change frame
            v_frame = int(max_acc_tracks['frame'])

            if v_dir == 2: # left-to-right direction
                if final_frame_lane > first_frame_lane: # max_acc_tracks['yAcceleration'] > 0:
                    action = 'right_lane_change'
                else:
                    action = 'left_lane_change'

            else:   # right-to-left direction
                if final_frame_lane > first_frame_lane: # max_acc_tracks['yAcceleration'] > 0:
                    action = 'left_lane_change'
                else:
                    action = 'right_lane_change'

            lane_change_info.append((v_id, desired_frame, v_dir, action))
        
        return lane_change_info

def str2num(action):
    if action == 'lane_keeping':
        return 0
    elif action == 'left_lane_change':
        return 1
    elif action == 'right_lane_change':
        return 2
    else:
        return 0
    
def num2str(action):
    if action == 0:
        return 'lane_keeping'
    elif action == 1:
        return 'left_lane_change'
    elif action == 2:
        return 'right_lane_change'
    else:
        return 'lane_keeping'

def desired_action_list(action):
    if type(action) is str:
        action = str2num(action)
    
    if action == 0:
        action_list = [1, 0, 0]
    elif action == 1:
        action_list = [0, 1, 0]
    elif action == 2:
        action_list = [0, 0, 1]
    else:
        action_list = [1, 0, 0]

    return action_list

def get_most_recent_file(dir=''):
    all_files = os.listdir(dir)
    files = [f for f in all_files if os.path.isfile(os.path.join(dir, f))]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(dir, x)), reverse=True)
    return files[0]