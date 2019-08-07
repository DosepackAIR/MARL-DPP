import settings
import random
import numpy as np
random.seed(settings.SEED)
np.random.seed(settings.SEED)
import datetime
import argparse
from src.configuration import Configuration
from copy import deepcopy
import tensorflow as tf
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# tf.set_random_seed(settings.SEED)


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputfile", required=True,
                    help="Give number of input file which you need to load before start training 0 will specify no loading of any input file")
    ap.add_argument("-o", "--outputfile", required=True,
                    help="Give number of output file where you need to store data after training is done 0 will specify no loading to any output file")
    ap.add_argument("-dd", "--detect_deadlock", required=True,
                    help="Tell whether you want to detect deadlock conditions before starting training or not")
    ap.add_argument("-bn", "--block_number", required=True,
                    help="Give block number")
    ap.add_argument("-l", "--layout_number", required=True,
                    help="Give layout number")
    ap.add_argument("-v", "--validate", required=True,
                    help="validate or train")

    args = vars(ap.parse_args())
    return args


def give_configuration_object(layout_number, block_number):
    layout_in_use = 'layout_' + str(layout_number)
    exec('from src.layout import ' + layout_in_use)
    environment_config = eval(layout_in_use + '[' + str(block_number) + ']')
    ed = Configuration(env_config=environment_config)
    return ed


def start_training(alg, input_file_number, validate, ed, dd, output_file_number):
    if validate:
        for i in ed.initial_states:
            alg.validate_given_state(1, input_file_number, '0', i[0], len(i[0]), i[1], i[2], 0)
    else:
        number_of_episode_for_deadlock_detect = 100
        start_time = datetime.datetime.now()
        print("start_time is ", start_time)
        if dd and hasattr(alg, 'detect_deadlocks_v0'):
            for num, i in enumerate(ed.initial_states):
                print('#################################################################')
                print(i, "block_list")
                if len(i[0]) == 1:
                    continue
                while True:
                    print("num", num, "state", i)
                    episode = alg.detect_deadlocks_v0(number_of_episode_for_deadlock_detect,
                                                  '0', '0', i[0], len(i[0]), i[1],
                                                  i[2])
                    print("episode", episode)
                    if not episode:
                        print("episode is False")
                        break
                    elif episode == (number_of_episode_for_deadlock_detect - 1):
                        break

        for num, state in enumerate(ed.initial_states):
            print('\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print("num", num, "state", state)
            alg.train_given_state(settings.EPISODES, input_file_number, output_file_number, state[0], len(state[0]), state[1], state[2])
        print("Total time taken is:", datetime.datetime.now()-start_time)


def main():
    params = deepcopy(settings.params)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.InteractiveSession(config=session_conf)

    args = parse_arguments()
    input_file_number = args["inputfile"]
    output_file_number = args["outputfile"]
    dd = int(args["detect_deadlock"])
    bn = int(args["block_number"])
    layout = int(args["layout_number"])
    validate = int(args["validate"])

    layout_number, block_number = layout, bn
    ed = give_configuration_object(layout_number, block_number)

    exec('from src.algorithms.' + settings.algorithm + ' import RL')
    params['sess'] = sess
    params['env_config'] = ed

    rl = eval('RL(**params)')
    start_training(rl, input_file_number, validate, ed, dd, output_file_number)


if __name__ == '__main__':
    main()

