% unsafe lane change actions
right_is_unsafe:- 
    right_is_invalid; right_is_busy.

left_is_unsafe:- 
    left_is_busy; left_is_invalid.

% dangerous lane change actions
right_is_dangerous:- 
    front_right_is_busy,front_right_velocity_is_lower,front_right_distance_is_unsafe; 
    back_right_is_busy,back_right_velocity_is_bigger,back_left_distance_is_unsafe.

left_is_dangerous:-
    front_left_is_busy,front_left_velocity_is_lower,front_left_distance_is_unsafe; 
    back_left_is_busy,back_left_velocity_is_bigger,back_left_distance_is_unsafe.
    
lane_keeping_is_better:- 
    back_is_busy,back_distance_is_unsafe,back_velocity_is_bigger.

% efficient lane change actions
right_is_better:- 
    front_right_is_free,front_left_is_busy,right_is_free,left_is_busy,front_is_busy.

left_is_better:- 
    front_is_busy,front_left_is_free,left_is_free.

% acceleration
reachDesiredSpeed:- front_is_free.

reachFrontSpeed:- 
    front_is_busy,front_distance_is_safe.

brake:- 
    front_is_busy,front_distance_is_unsafe.

