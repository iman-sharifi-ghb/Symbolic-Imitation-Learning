from third_party.popper.util import Settings, print_prog_score
from third_party.popper.loop import learn_solution
from utils import create_setting_bk_example
from pyswip import Prolog
import time
import warnings
warnings.filterwarnings("ignore")
prolog = Prolog()

def reset_prolog():

    predicates = list(prolog.query("findall(Pred, current_predicate(Pred/Arity), Predicates)"))
    predicate_list = predicates[0]['Predicates']
    # print(len(predicate_list))
    print(len(predicate_list))
    # for pred in predicate_list:
    #     print(str(pred)+"(_)")
        # prolog.assertz(f":- dynamic {str(pred)}/1")
        # prolog.retractall(f"{str(pred)}(_)")
        


    # bk_pred_list1 = ['front_is_busy','front_right_is_busy','front_right_is_busy',\
    #                 'right_is_busy','back_right_is_busy','back_is_busy','back_left_is_busy',\
    #                 'left_is_busy','front_left_is_busy']     
    # bk_pred_list2 = ['front_is_free','front_right_is_free','front_right_is_free',\
    #                 'right_is_free','back_right_is_free','back_is_free','back_left_is_free',\
    #                 'left_is_free','front_left_is_free']
    # bk_pred_list3 = ['right_is_invalid','right_is_valid','left_is_invalid','left_is_valid']
    # bk_pred_list4 = ['ego_velocity_is_legal','ego_velocity_is_illegal',\
    #                     'front_distance_is_safe','front_distance_is_critical']
    # bk_pred_list5 = ['front_velocity_is_bigger','front_left_velocity_is_bigger','front_right_velocity_is_bigger',\
    #                     'back_left_velocity_is_bigger','back_right_velocity_is_bigger']
    # bk_pred_list5 = ['front_velocity_is_lower','front_left_velocity_is_lower','front_right_velocity_is_lower',\
    #                     'back_left_velocity_is_lower','back_right_velocity_is_lower']
    
    # bk_pred_list = bk_pred_list1 + bk_pred_list2 + bk_pred_list3 + bk_pred_list4 + bk_pred_list5

    # for pred in bk_pred_list:
    #     prolog.retractall(pred+"(_)")


    # prolog.retractall('head_pred(_)')
    # prolog.retractall('body_pred(_)')
    # prolog.retractall('pos(_)')
    # prolog.retractall('neg(_)')

if __name__ == '__main__':
    f1 = 'bias.pl'
    f2 = 'bk.pl'
    f3 = 'exs.pl'

    # write the desired rules to a file
    f_name = 'extracted_rules.pl'
    # L = [i+1 for i in range(4)]
    L = [14]
    
    for i in L:

        Dir = 'SIL/data/'
        if i == 1:
            dir = 'unsafe/right/'
        elif i == 2:
            dir = 'unsafe/left/'
        elif i == 3:
            dir = 'dangerous/right/'
        elif i == 4:
            dir = 'dangerous/left/'
        elif i == 41:
            dir = 'dangerous/keep/'
        elif i == 5:
            dir = 'efficiency/right/'
        elif i == 6:
            dir = 'efficiency/left/'
        elif i == 7:
            dir = 'velocity/increase/'
        elif i == 8:
            dir = 'velocity/decrease/'
        elif i == 9:
            dir = 'acceleration/desiredSpeed/'
        elif i == 10:
            dir = 'acceleration/frontSpeed/'
        elif i == 11:
            dir = 'acceleration/brake/'
        elif i == 12:
            dir = 'augmented data/unsafe/right/'
        elif i == 13:
            dir = 'augmented data/unsafe/left/'
        elif i == 14:
            dir = 'unsafe/noisy/right/'
        else:
            dir = ''
        
        # create_setting_bk_example(rule=i, bias_file=Dir+dir+f1, bk_file=Dir+dir+f2, ex_file=Dir+dir+f3)
        dir = 'acceleration/desiredSpeed2/'
        t0 = time.time()
        settings = Settings(cmd_line=False, quiet=False, bias_file=Dir+dir+f1, bk_file=Dir+dir+f2, ex_file=Dir+dir+f3)
        prog, score, stats = learn_solution(settings)
        if prog != None:
            extracted_rule = print_prog_score(prog, score)
        else:
            print('NO SOLUTION')
            extracted_rule = None

        sections = ['front', 'front_right', 'right', 'back_right', 'back', 'back_left', 'left', 'front_left']
        if extracted_rule:
            extracted_rule = extracted_rule.replace("(A)","")
            for section in sections:
                extracted_rule = extracted_rule.replace(section+'_is_free',"not("+section+'_is_busy)')
            extracted_rule += "\n"
            file_name = Dir+dir+f_name
            with open(file_name, 'w') as f:
                f.write(extracted_rule)
            f.close()
        if settings.show_stats:
            stats.show()

        t = time.time() - t0
        print(f'Elapsed time: {t}')

        # reset_prolog()

        time.sleep(1)

        
