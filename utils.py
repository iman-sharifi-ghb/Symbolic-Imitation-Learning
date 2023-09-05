import random


def create_setting_bk_example(rule=1, bias_file="bias.pl", bk_file="bk.pl", ex_file="exs.pl"):

    if rule == 1:
        right_unsafe_rule = True
    else:
        right_unsafe_rule = False

    if rule == 2:
        left_unsafe_rule = True
    else:
        left_unsafe_rule = False

    if rule == 3:
        right_dangerous_rule = True
    else:
        right_dangerous_rule = False

    if rule == 4:
        left_dangerous_rule = True
    else:
        left_dangerous_rule = False

    if rule == 41:
        keep_efficiency_rule = True
    else:
        keep_efficiency_rule = False

    if rule == 5:
        right_efficiency_rule = True
    else:
        right_efficiency_rule = False

    if rule == 6:
        left_efficiency_rule = True
    else:
        left_efficiency_rule = False

    if rule == 7:
        increase_velocity_rule = True
    else:
        increase_velocity_rule = False

    if rule == 8:
        decrease_velocity_rule = True
    else:
        decrease_velocity_rule = False

    if rule == 9:
        acceleration_rule1 = True
    else:
        acceleration_rule1 = False

    if rule == 10:
        acceleration_rule2 = True
    else:
        acceleration_rule2 = False

    if rule == 11:
        acceleration_rule3 = True
    else:
        acceleration_rule3 = False

    state = ['busy', 'free']
    validity = ['valid', 'invalid']
    velocity = ['legal', 'illegal']
    distance = ['safe', 'critical']
    comparison = ['lower', 'equal', 'bigger']

    with open(bias_file, 'w') as f_b:

        # turn off the warnings
        f_b.write(":- style_check(-discontiguous).\n")
        # f_b.write(":- style_check(-singleton).\n")

        # f_b.write(":- dynamic head_pred/1, body_pred/1.\n")
        # f_b.write(":- abolish(head_pred/1).\n")
        # f_b.write(":- abolish(body_pred/1).\n\n")
        # f_b.write(":- retractall(head_pred(_)).\n")
        # f_b.write(":- retractall(body_pred(_)).\n\n")

        # writing Head predicate in the bias file
        if right_unsafe_rule:
            f_b.write("head_pred(right_is_unsafe,1).\n\n")

        if left_unsafe_rule:
            f_b.write("head_pred(left_is_unsafe,1).\n\n")

        if right_dangerous_rule:
            f_b.write("head_pred(right_is_dangerous,1).\n\n")

        if left_dangerous_rule:
            f_b.write("head_pred(left_is_dangerous,1).\n\n")

        if keep_efficiency_rule:
            f_b.write("head_pred(lane_keeping_is_better,1).\n\n")

        if right_efficiency_rule:
            f_b.write("head_pred(right_is_better,1).\n\n")

        if left_efficiency_rule:
            f_b.write("head_pred(left_is_better,1).\n\n")

        if increase_velocity_rule:
            f_b.write("head_pred(increase_velocity,1).\n\n")

        if decrease_velocity_rule:
            f_b.write("head_pred(decrease_velocity,1).\n\n")

        if acceleration_rule1:
            f_b.write("head_pred(reachDesiredSpeed,1).\n\n")

        if acceleration_rule2:
            f_b.write("head_pred(reachFrontSpeed,1).\n\n")

        if acceleration_rule3:
            f_b.write("head_pred(brake,1).\n\n")    

        # writing Body predicates in the bias file
        f_b.write("body_pred(front_is_busy,1).\n")
        f_b.write("body_pred(front_right_is_busy,1).\n")
        f_b.write("body_pred(right_is_busy,1).\n")
        f_b.write("body_pred(back_right_is_busy,1).\n")
        f_b.write("body_pred(back_is_busy,1).\n")
        f_b.write("body_pred(back_left_is_busy,1).\n")
        f_b.write("body_pred(left_is_busy,1).\n")
        f_b.write("body_pred(front_left_is_busy,1).\n")

        f_b.write("body_pred(front_is_free,1).\n")
        f_b.write("body_pred(front_right_is_free,1).\n")
        f_b.write("body_pred(right_is_free,1).\n")
        f_b.write("body_pred(back_right_is_free,1).\n")
        f_b.write("body_pred(back_is_free,1).\n")
        f_b.write("body_pred(back_left_is_free,1).\n")
        f_b.write("body_pred(left_is_free,1).\n")
        f_b.write("body_pred(front_left_is_free,1).\n")

        # if rule <= 4:
        f_b.write("body_pred(right_is_valid,1).\n")
        f_b.write("body_pred(left_is_valid,1).\n")
        f_b.write("body_pred(right_is_invalid,1).\n")
        f_b.write("body_pred(left_is_invalid,1).\n")

        # if rule >= 7:
        f_b.write("body_pred(ego_velocity_is_legal,1).\n")
        f_b.write("body_pred(ego_velocity_is_illegal,1).\n")
        f_b.write("body_pred(front_distance_is_safe,1).\n")
        f_b.write("body_pred(front_distance_is_critical,1).\n")
        f_b.write("body_pred(back_distance_is_safe,1).\n")
        f_b.write("body_pred(back_distance_is_critical,1).\n")


        f_b.write("body_pred(front_velocity_is_bigger,1).\n")
        f_b.write("body_pred(front_velocity_is_equal,1).\n")
        f_b.write("body_pred(front_velocity_is_lower,1).\n")

        f_b.write("body_pred(back_velocity_is_bigger,1).\n")
        f_b.write("body_pred(back_velocity_is_equal,1).\n")
        f_b.write("body_pred(back_velocity_is_lower,1).\n")

        f_b.write("body_pred(front_left_velocity_is_lower,1).\n")
        f_b.write("body_pred(front_left_velocity_is_equal,1).\n")
        f_b.write("body_pred(front_left_velocity_is_bigger,1).\n")

        f_b.write("body_pred(back_left_velocity_is_lower,1).\n")
        f_b.write("body_pred(back_left_velocity_is_equal,1).\n")
        f_b.write("body_pred(back_left_velocity_is_bigger,1).\n")

        f_b.write("body_pred(front_right_velocity_is_lower,1).\n")
        f_b.write("body_pred(front_right_velocity_is_equal,1).\n")
        f_b.write("body_pred(front_right_velocity_is_bigger,1).\n")

        f_b.write("body_pred(back_right_velocity_is_lower,1).\n")
        f_b.write("body_pred(back_right_velocity_is_equal,1).\n")
        f_b.write("body_pred(back_right_velocity_is_bigger,1).\n")
    f_b.close()
        
    N = 0
    with open(bk_file, 'w') as f_bk:

        # turn off the warnings
        f_bk.write(":- style_check(-discontiguous).\n\n")

        bk_pred_list1 = ['front_is_busy','front_right_is_busy','front_right_is_busy',\
                        'right_is_busy','back_right_is_busy','back_is_busy','back_left_is_busy',\
                        'left_is_busy','front_left_is_busy']     
        bk_pred_list2 = ['front_is_free','front_right_is_free','front_right_is_free',\
                        'right_is_free','back_right_is_free','back_is_free','back_left_is_free',\
                        'left_is_free','front_left_is_free']
        bk_pred_list3 = ['right_is_invalid','right_is_valid','left_is_invalid','left_is_valid']
        bk_pred_list4 = ['ego_velocity_is_legal','ego_velocity_is_illegal',\
                         'front_distance_is_safe','front_distance_is_critical']
        bk_pred_list5 = ['front_velocity_is_bigger','front_left_velocity_is_bigger','front_right_velocity_is_bigger',\
                         'back_left_velocity_is_bigger','back_right_velocity_is_bigger']
        bk_pred_list6 = ['front_velocity_is_equal','front_left_velocity_is_equal','front_right_velocity_is_equal',\
                         'back_left_velocity_is_equal','back_right_velocity_is_equal']
        bk_pred_list7 = ['front_velocity_is_lower','front_left_velocity_is_lower','front_right_velocity_is_lower',\
                         'back_left_velocity_is_lower','back_right_velocity_is_lower']
        
        bk_pred_list = bk_pred_list1 + bk_pred_list2 + bk_pred_list3 + bk_pred_list4 + bk_pred_list5 + bk_pred_list6 + bk_pred_list7

        # for pred in bk_pred_list:
        #     F = f":- dynamic {pred}/1.\n"
        #     f_bk.write(F)

        # for pred in bk_pred_list:
        #     # F = f":- abolish({pred}/1).\n"
        #     F = f":- retractall({pred}(_)).\n"
        #     f_bk.write(F)

        # Background Knowledge
        f_bk.write("front_is_free(X):-not(front_is_busy(X)),!.\n")
        f_bk.write("front_right_is_free(X):-not(front_right_is_busy(X)),!.\n")
        f_bk.write("right_is_free(X):-not(right_is_busy(X)),!.\n")
        f_bk.write("back_right_is_free(X):-not(back_right_is_busy(X)),!.\n")
        f_bk.write("back_is_free(X):-not(back_is_busy(X)),!.\n")
        f_bk.write("back_left_is_free(X):-not(back_left_is_busy(X)),!.\n")
        f_bk.write("left_is_free(X):-not(left_is_busy(X)),!.\n")
        f_bk.write("front_left_is_free(X):-not(front_left_is_busy(X)),!.\n\n")

        f_bk.write("right_is_invalid(X):-not(right_is_valid(X)),!.\n")
        f_bk.write("left_is_invalid(X):-not(left_is_valid(X)),!.\n")
        f_bk.write("ego_velocity_is_legal(X):-not(ego_velocity_is_illegal(X)),!.\n")
        f_bk.write("front_distance_is_safe(X):-not(front_distance_is_critical(X)),!.\n")

        # f_bk.write("front_velocity_is_bigger(X):-not(front_velocity_is_lower(X)),!.\n")
        # f_bk.write("front_left_velocity_is_bigger(X):-not(front_left_velocity_is_lower(X)),!.\n")
        # f_bk.write("front_right_velocity_is_bigger(X):-not(front_right_velocity_is_lower(X)),!.\n")
        # f_bk.write("back_left_velocity_is_bigger(X):-not(back_left_velocity_is_lower(X)),!.\n")
        # f_bk.write("back_right_velocity_is_bigger(X):-not(back_right_velocity_is_lower(X)),!.\n\n")

        # define all possible states (consisting of business and validity of each section around the AV)
        with open(ex_file, 'w') as f_ex:
            # turn off the warnings
            f_ex.write(":- style_check(-discontiguous).\n")
            # f_ex.write(":- dynamic pos/1, neg/1.\n")
            # f_ex.write(":- abolish(pos/1).\n")
            # f_ex.write(":- abolish(neg/1).\n\n")
            # f_ex.write(":- retractall(pos(_)).\n")
            # f_ex.write(":- retractall(neg(_)).\n\n")

            for v1 in validity:
                for v2 in validity:
                    for s1 in state:
                        for s2 in state:
                            for s3 in state:
                                for s4 in state:
                                    for s5 in state:
                                        for s6 in state:
                                            for s7 in state:
                                                for s8 in state:
                                                    N += 1
                                                    f0 = f"% scenario no. {N}:\n"
                                                    # f00 = f"state(s{N}).\n"
                                                    f1 = f"front_is_busy(s{N}).\n"
                                                    f2 = f"front_right_is_busy(s{N}).\n"
                                                    f3 = f"right_is_busy(s{N}).\n"
                                                    f4 = f"back_right_is_busy(s{N}).\n"
                                                    f5 = f"back_is_busy(s{N}).\n"
                                                    f6 = f"back_left_is_busy(s{N}).\n"
                                                    f7 = f"left_is_busy(s{N}).\n"
                                                    f8 = f"front_left_is_busy(s{N}).\n"

                                                    f1_1 = f"front_is_free(s{N}).\n"
                                                    f2_1 = f"front_right_is_free(s{N}).\n"
                                                    f3_1 = f"right_is_free(s{N}).\n"
                                                    f4_1 = f"back_right_is_free(s{N}).\n"
                                                    f5_1 = f"back_is_free(s{N}).\n"
                                                    f6_1 = f"back_left_is_free(s{N}).\n"
                                                    f7_1 = f"left_is_free(s{N}).\n"
                                                    f8_1 = f"front_left_is_free(s{N}).\n"

                                                    f9 = f"right_is_valid(s{N}).\n"
                                                    f9_1 = f"right_is_invalid(s{N}).\n"
                                                    f10 = f"left_is_valid(s{N}).\n"
                                                    f10_1 = f"left_is_invalid(s{N}).\n"

                                                    velocity_permision = random.choice(velocity)
                                                    if velocity_permision == 'legal':
                                                        f11 = f"ego_velocity_is_legal(s{N}).\n"
                                                    else:
                                                        f11 = f"ego_velocity_is_illegal(s{N}).\n"
                                                    
                                                    if s1 == 'free':
                                                        f12 = f"front_distance_is_safe(s{N}).\n"
                                                        front_distance_permision = 'safe'
                                                    else:
                                                        front_distance_permision = random.choice(distance)
                                                        if front_distance_permision == 'safe':
                                                            f12 = f"front_distance_is_safe(s{N}).\n"
                                                        else:
                                                            f12 = f"front_distance_is_critical(s{N}).\n"

                                                    velocity_comparison = random.choice(comparison)
                                                    if velocity_comparison == 'lower':
                                                        f13 = f"front_velocity_is_lower(s{N}).\n"
                                                    elif velocity_comparison == 'equal':
                                                        f13 = f"front_velocity_is_equal(s{N}).\n"
                                                    else:
                                                        f13 = f"front_velocity_is_bigger(s{N}).\n"

                                                    velocity_comparison1 = random.choice(comparison)
                                                    if velocity_comparison1 == 'lower':
                                                        f14 = f"front_left_velocity_is_lower(s{N}).\n"
                                                    elif velocity_comparison1 == 'equal':
                                                        f14 = f"front_left_velocity_is_equal(s{N}).\n"
                                                    else:
                                                        f14 = f"front_left_velocity_is_bigger(s{N}).\n"

                                                    velocity_comparison2 = random.choice(comparison)
                                                    if velocity_comparison2 == 'lower':
                                                        f15 = f"front_right_velocity_is_lower(s{N}).\n"
                                                    elif velocity_comparison2 == 'equal':
                                                        f15 = f"front_right_velocity_is_equal(s{N}).\n"
                                                    else:
                                                        f15 = f"front_right_velocity_is_bigger(s{N}).\n"

                                                    velocity_comparison3 = random.choice(comparison)
                                                    if velocity_comparison3 == 'lower':
                                                        f16 = f"back_left_velocity_is_lower(s{N}).\n"
                                                    elif velocity_comparison3 == 'equal':
                                                        f16 = f"back_left_velocity_is_equal(s{N}).\n"
                                                    else:
                                                        f16 = f"back_left_velocity_is_bigger(s{N}).\n"

                                                    velocity_comparison4 = random.choice(comparison)
                                                    if velocity_comparison4 == 'lower':
                                                        f17 = f"back_right_velocity_is_lower(s{N}).\n\n"
                                                    elif velocity_comparison4 == 'equal':
                                                        f17 = f"back_right_velocity_is_equal(s{N}).\n\n"
                                                    else:
                                                        f17 = f"back_right_velocity_is_bigger(s{N}).\n\n"

                                                    velocity_comparison5 = random.choice(comparison)
                                                    if velocity_comparison5 == 'lower':
                                                        f18 = f"back_velocity_is_lower(s{N}).\n"
                                                    elif velocity_comparison5 == 'equal':
                                                        f18 = f"back_velocity_is_equal(s{N}).\n"
                                                    else:
                                                        f18 = f"back_velocity_is_bigger(s{N}).\n"

                                                    if s5 == 'free':
                                                        f19 = f"back_distance_is_safe(s{N}).\n"
                                                        back_distance_permision = 'safe'
                                                    else:
                                                        back_distance_permision = random.choice(distance)
                                                        if back_distance_permision == 'safe':
                                                            f19 = f"back_distance_is_safe(s{N}).\n"
                                                        else:
                                                            f19 = f"back_distance_is_critical(s{N}).\n"

                                                    # ===========================================================    
                                                    # Write facts in each state
                                                    f_bk.write(f0); #f_bk.write(f00)

                                                    # business of each section aroung the AV
                                                    if s1 == 'busy':
                                                        f_bk.write(f1)
                                                    else:
                                                        f_bk.write(f1_1)
                                                    
                                                    if s2 == 'busy':
                                                        f_bk.write(f2)
                                                    else:
                                                        f_bk.write(f2_1)

                                                    if s3 == 'busy':
                                                        f_bk.write(f3)
                                                    else:
                                                        f_bk.write(f3_1)

                                                    if s4 == 'busy':
                                                        f_bk.write(f4)
                                                    else:
                                                        f_bk.write(f4_1)

                                                    if s5 == 'busy':
                                                        f_bk.write(f5)
                                                    else:
                                                        f_bk.write(f5_1)

                                                    if s6 == 'busy':
                                                        f_bk.write(f6)
                                                    else:
                                                        f_bk.write(f6_1)

                                                    if s7 == 'busy':
                                                        f_bk.write(f7)
                                                    else:
                                                        f_bk.write(f7_1)

                                                    if s8 == 'busy':
                                                        f_bk.write(f8)
                                                    else:
                                                        f_bk.write(f8_1)

                                                    # Write validity rules for lane changes
                                                    if v1 == 'valid':
                                                        f_bk.write(f9)
                                                    else:
                                                        f_bk.write(f9_1)
                                                        
                                                    if v2 == 'valid':
                                                        f_bk.write(f10)
                                                    else:
                                                        f_bk.write(f10_1)

                                                    # write velocity and distance permisions 
                                                    f_bk.write(f11)
                                                    f_bk.write(f12)
                                                    f_bk.write(f19)
                                                    # if s1 == 'busy':
                                                    f_bk.write(f13)

                                                    # if s8 == 'busy':
                                                    f_bk.write(f14)

                                                    # if s2 == 'busy':
                                                    f_bk.write(f15)

                                                    # if s6 == 'busy':
                                                    f_bk.write(f16)
                                                    
                                                    # if s4 == 'busy':
                                                    f_bk.write(f17)

                                                    f_bk.write(f18)
                                                    
                                                    # ================================================================
                                                    # Examples

                                                    # Absolute safety examples
                                                    if right_unsafe_rule:
                                                        if s3 == "busy" or v1 == "invalid":
                                                            F = f"pos(right_is_unsafe(s{N})).\n"
                                                            f_ex.write(F)                                                
                                                        else:
                                                            F = f"neg(right_is_unsafe(s{N})).\n"
                                                            f_ex.write(F)

                                                    if left_unsafe_rule:
                                                        if s7 == "busy" or v2 == "invalid":
                                                            F = f"pos(left_is_unsafe(s{N})).\n"
                                                            f_ex.write(F)
                                                        else:
                                                            F = f"neg(left_is_unsafe(s{N})).\n"
                                                            f_ex.write(F)

                                                    # Partial safety examples
                                                    if right_dangerous_rule:
                                                        if (s2 == "busy" and velocity_comparison2 == 'lower') or (s4 =="busy" and velocity_comparison4 == 'bigger'):
                                                            F = f"pos(right_is_dangerous(s{N})).\n"
                                                            f_ex.write(F)                                                
                                                        else:
                                                            F = f"neg(right_is_dangerous(s{N})).\n"
                                                            f_ex.write(F)

                                                    if left_dangerous_rule:
                                                        if (s8 == "busy" and velocity_comparison1 == 'lower') or (s6 =="busy" and velocity_comparison3 == 'bigger'):
                                                            F = f"pos(left_is_dangerous(s{N})).\n"
                                                            f_ex.write(F)
                                                        else:
                                                            F = f"neg(left_is_dangerous(s{N})).\n"
                                                            f_ex.write(F)

                                                    if keep_efficiency_rule:
                                                        if (s5 == "busy" and velocity_comparison5 == 'bigger' and back_distance_permision == 'critical'):
                                                            F = f"pos(lane_keeping_is_better(s{N})).\n"
                                                            f_ex.write(F)
                                                        else:
                                                            F = f"neg(lane_keeping_is_better(s{N})).\n"
                                                            f_ex.write(F)
                                                    
                                                    # Efficiency examples
                                                    if left_efficiency_rule:
                                                        if s1 == "busy" and s8 =="free" and s7 == "free":
                                                            F = f"pos(left_is_better(s{N})).\n"
                                                            f_ex.write(F)
                                                        else:
                                                            F = f"neg(left_is_better(s{N})).\n"
                                                            f_ex.write(F)

                                                    if right_efficiency_rule:
                                                        if s1 == "busy" and s8 =="busy" and s7 == "busy" and s2 == "free" and s3 == "free":
                                                            F = f"pos(right_is_better(s{N})).\n"
                                                            f_ex.write(F)
                                                        else:
                                                            F = f"neg(right_is_better(s{N})).\n"
                                                            f_ex.write(F)
                                                    
                                                    # Velocity examples
                                                    if increase_velocity_rule:
                                                        if (velocity_permision == 'legal' and s1 == 'free') or (velocity_permision == 'legal' and s1 == 'busy' and front_distance_permision == 'safe' and velocity_comparison == 'bigger'):
                                                            F = f"pos(increase_velocity(s{N})).\n"
                                                            f_ex.write(F)
                                                        else:
                                                            F = f"neg(increase_velocity(s{N})).\n"
                                                            f_ex.write(F)

                                                    if decrease_velocity_rule:
                                                        if (velocity_permision == 'illegal') or (velocity_permision == 'legal' and s1 == 'busy' and front_distance_permision == 'safe' and velocity_comparison == 'lower') or (velocity_permision == 'legal' and s1 == 'busy' and front_distance_permision == 'critical'):
                                                            F = f"pos(decrease_velocity(s{N})).\n"
                                                            f_ex.write(F)
                                                        else:
                                                            F = f"neg(decrease_velocity(s{N})).\n"
                                                            f_ex.write(F)
                                                    
                                                    # Acceleration and deceleration examples
                                                    if acceleration_rule1:
                                                        if s1 == 'free':
                                                            F = f"pos(reachDesiredSpeed(s{N})).\n"
                                                            f_ex.write(F)
                                                        else: #velocity_permision == 'illegal' or s1 == 'busy':
                                                            F = f"neg(reachDesiredSpeed(s{N})).\n"
                                                            f_ex.write(F)
                                                    

                                                    if acceleration_rule2:
                                                        if s1 == 'busy' and front_distance_permision == 'safe':
                                                            F = f"pos(reachFrontSpeed(s{N})).\n"
                                                            f_ex.write(F)
                                                        else:
                                                            F = f"neg(reachFrontSpeed(s{N})).\n"
                                                            f_ex.write(F)


                                                    if acceleration_rule3:
                                                        if s1 == 'busy' and front_distance_permision == 'critical':
                                                            F = f"pos(brake(s{N})).\n"
                                                            f_ex.write(F)
                                                        else: #velocity_permision == 'legal' and s1 == 'free':
                                                            F = f"neg(brake(s{N})).\n"
                                                            f_ex.write(F)
                                                    
            f_ex.close()
                                           
        f_bk.close()
    

