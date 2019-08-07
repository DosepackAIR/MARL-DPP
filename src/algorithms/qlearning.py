import cv2
import numpy as np
import time
from copy import deepcopy
import random
import itertools as it
import re
from src.utils.utils import *
from collections import OrderedDict
from src.env_gym import Environment


class RL:
    def __init__(self, input_dim, output_dim, cars, sess, env_config,
                 learning_rate=0.0001, batch_size=32):
        self.env_config = env_config
        global env
        env = Environment(self.env_config)
        self.set_finished_box = ""
        self.epsilon = 0
        self.alpha = 0.9
        self.discount = 0.15
        self.cars_reached_destination = 0
        self.episode_iterations = 20
        self.episode_array = []
        self.probability_model = 3
        self.discount_model = 1
        self.training_done = 0
        self.validation_done = 0
        self.block_destination_count = 0
        self.training_complete_destination_index = self.env_config.training_complete_destination_index
        # self.one_to_one_car_mapping()  # TODO: review its usage

    def getqvalue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state, action) not in self.qvals:
            self.qvals[(state, action)] = 0.0
        return self.qvals[(state, action)]

    def computevaluefromqvalues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalactions = env.getlegalactions(deepcopy(env.state_to_array(state)))
        if len(legalactions) == 0:
            return 0.0
        tmp = Counter()
        for action in legalactions:
            tmp[action] = self.getqvalue(state, action)
        return tmp[tmp.argMax()]

    def computeactionfromqvalues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalactions = env.getlegalactions(env.state_to_array(state))
        if len(legalactions) == 0:
            return None
        tmp = Counter()
        for action in legalactions:
            tmp[action] = self.getqvalue(state, action)
        return tmp.argMax()

    def getaction(self, state, epsilon):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalactions = env.getlegalactions(env.state_to_array(state))

        # if self.epsilon == 0:
        #     for action in legalactions:
        #         print(self.qvals[(state, action)])
        #     print(state, legalactions)
        #     input()
        # print("legal actions", legalactions, "for state", state)
        # print ("for_given_state", state, "legal_actions are", legalactions)
        # print ("legalactions",legalactions,"state",state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalactions) != 0:
            if flipcoin(epsilon):
                # if not epsilon:
                # print ("random")
                action = random.choice(legalactions)
            else:
                action = self.computeactionfromqvalues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here


        """
        self.qvals[(state, action)] = self.qvals[(state, action)] + self.alpha * (
                    reward + self.discount * self.computevaluefromqvalues(nextState) - self.qvals[(state, action)])

    def getpolicy(self, state):
        return self.computeactionfromqvalues(state)

    def getvalue(self, state):
        return self.computevaluefromqvalues(state)

    # print(state_to_array("D5SxD4WzD2xD4zD3xD4"))
    # def get_reward(self, previous_state, state, car_id, action='S'):
    #     """
    #     This function calculates the instantaneous reward that car will get performing this state transistion
    #     :param self : Default parameter for initialized objects
    #     :param previous_state: Previous state of the car
    #     :param state: new state for the car
    #     :param car_id: id of the car
    #     :param action: action which leads to state from previous_state
    #     :return : reward
    #     """
    #     state_array = self.state_to_array(state)
    #     previous_state_array = self.state_to_array(previous_state)
    #     reward = -10
    #     # if action != 'S':
    #     #    reward -= 10
    #     # =============================================================================
    #     #         state == destination
    #     # =============================================================================
    #     if state_array[0][0] == state_array[0][1]:
    #         reward += 10010
    #     # =============================================================================
    #     # Checking the box of other cars == box of present car after performing this transistion AND if they are on
    #     # same floor; then give heavy negative reward
    #     # =============================================================================
    #     for num, i in enumerate(self.cars):
    #         if num == car_id:
    #             continue
    #         state_box = int(re.search(r'\d+', state_array[0][0]).group())
    #         car_box = int(re.search(r'\d+', i.source_node).group())
    #         if (state_box == car_box) and (state_array[0][0][0] == i.source_node[0]):
    #             # print("car_id",car_id,"state_box",state_box,"car_box",car_box,"car_num",num,"i.source_node",i.source_node,"array00",array[0][0])
    #             reward -= 10000
    #     return reward

    def save_qvalues(self, outputfile_number):
        file_name = "./src/data/qvalue_files/layout-3/" + outputfile_number + ".txt"
        if outputfile_number == '0':
            return None
        with open(file_name, 'w+') as f:
            for i, j in self.qvals.keys():
                string = str(i) + '|' + j + '|' + repr(self.qvals[(i, j)]) + '\n'
                f.write(string)
                # json.dump(self.qvals,f)

    def load_qvalues(self, inputfile_number):
        file_name = "./src/data/qvalue_files/layout-3/" + inputfile_number + ".txt"
        self.qvals = Counter()
        if inputfile_number == '0':
            return None
        with open(file_name, 'r') as f:
            for i in f.read().split('\n'):
                if i == '':
                    continue
                state = i.split('|')[0]
                action = i.split('|')[1]
                value = i.split('|')[2]
                self.qvals[(state, action)] = float(value)
                # print ('qvalue', self.qvals[(state,action)])
                # print (state, action, value)
                # string = str(i)+'|'+j+'|'+str(self.qvals[(i,j)])+'\n'
                # f.write(string)

    def train_given_state(self, episodes, inputfile_number, outputfile_number, input_car_states=None, no_of_cars=1,
                          destination_index_array=[0, 0, 0, 0], destination_choose_arrays=[0, 0, 0]):

        self.load_qvalues(inputfile_number)
        for episode in range(0, episodes):
            total_reward = 0
            # print(env.blocked_nodes_list)
            if episode == 30:
                self.save_qvalues(outputfile_number)
            self.epsilon = give_action_choose_probability(episodes, episode, self.probability_model)
            env.reset()
            env.initialize_cars(no_of_cars)
            # print(env.blocked_nodes_list)
            """
            Defining initial sources and destinations for cars.
            We are also storing defined source into 'given_nodes' list.
            Which will be used later in exploitation part.
            """
            env.initialize(destination_index_array, input_car_states, destination_choose_arrays)
            env.select_reward_function('q_reward')
            print('episode', episode, self.epsilon)
            self.training_done = 0
            counts_in_one_episode = 0
            self.episode_array = []
            while not self.training_done:
                counts_in_one_episode += 1
                # self.random_shuffle_car_map()
                for num, i in enumerate(env.get_cars()):
                    # num0 = self.car_one_to_one_map_dict[num]
                    if env.check_car_training(num):
                        continue
                    state = env.encodestate(num, 1)
                    action = self.getaction(state, self.epsilon)
                    # print("state", state, "action", action)
                    self.episode_array.append(action)
                    next_state = env.get_successor_state(deepcopy(state), action)
                    reward = env.reward_function(deepcopy(state), deepcopy(next_state), num, action)
                    if self.epsilon <= 0:
                        print(state, action, next_state, reward)
                    # print(self.qvals[(state, action)])
                    self.update(state, action, next_state, reward)
                    next_state_array = env.state_to_array(next_state)
                    env.update_car(num, deepcopy(next_state_array[0][0]), deepcopy(next_state_array[0][1]), 1)
                    if self.epsilon <= 0:
                        env.update_simulator_display(10)
                    total_reward += reward
                self.training_done = env.is_game_ended()
            print('counts_in_one_episode', counts_in_one_episode)
            # print(total_reward)
            # env.initialize_simulator_variables()
            self.alpha = 0.9
            self.cars_reached_destination = 0
        self.save_qvalues(outputfile_number)

    def train_given_state_random_order(self, episodes, inputfile_number, outputfile_number, input_car_states=None, no_of_cars=1,
                          destination_index_array=[0, 0, 0, 0], destination_choose_arrays=[0, 0, 0]):

        self.load_qvalues(inputfile_number)
        for episode in range(0, episodes):
            total_reward = 0
            # print(env.blocked_nodes_list)
            if episode == 30:
                self.save_qvalues(outputfile_number)
            self.epsilon = give_action_choose_probability(episodes, episode, self.probability_model)
            env.reset()
            env.initialize_cars(no_of_cars)
            # print(env.blocked_nodes_list)
            """
            Defining initial sources and destinations for cars.
            We are also storing defined source into 'given_nodes' list.
            Which will be used later in exploitation part.
            """
            env.initialize(destination_index_array, input_car_states, destination_choose_arrays)
            env.select_reward_function('q_reward')
            print('episode', episode, self.epsilon)
            self.training_done = 0
            counts_in_one_episode = 0
            self.episode_array = []
            while not self.training_done:
                counts_in_one_episode += 1
                num = random.randint(0, env.no_of_cars - 1)
                if env.check_car_training(num):
                    continue
                state = env.encodestate(num, 1)
                action = self.getaction(state, self.epsilon)
                # print("state", state, "action", action)
                self.episode_array.append(action)
                if len(self.episode_array) > 10000:
                    # Added this for non-solvable states
                    print('\nSkipping! The state :: %s cannot be trained as it has no paths possible.' % state)
                    return 0
                next_state = env.get_successor_state(deepcopy(state), action)
                reward = env.reward_function(deepcopy(state), deepcopy(next_state), num, action)
                if self.epsilon <= 0:
                    print(state, action, next_state, reward)
                # print(self.qvals[(state, action)])
                self.update(state, action, next_state, reward)
                next_state_array = env.state_to_array(next_state)
                env.update_car(num, deepcopy(next_state_array[0][0]), deepcopy(next_state_array[0][1]), 1)
                if self.epsilon <= 0:
                    env.update_simulator_display(10)
                total_reward += reward
                self.training_done = env.is_game_ended()
            print('counts_in_one_episode', counts_in_one_episode)
            # print(total_reward)
            # env.initialize_simulator_variables()
            self.alpha = 0.9
            self.cars_reached_destination = 0
        self.save_qvalues(outputfile_number)

    def detect_deadlocks_v0(self, episodes=40, inputfile_number=0, outputfile_number=0, input_car_states=None,
                            no_of_cars=1, destination_index_array=[0, 0, 0, 0], destination_choose_arrays=[0, 0, 0],
                            max_cars=6):
        """
        Here we try and detect potential deadlocks that can occur while training a state. We have various criteria for
        classifying any state as a deadlocked state based on the number of consecutive stop actions that are given by
        the policy, if it takes more than 10,000 actions to clear a state etc.
        :param episodes:
        :param inputfile_number:
        :param outputfile_number:
        :param input_car_states:
        :param no_of_cars:
        :param destination_index_array:
        :param destination_choose_arrays:
        :return:
        """
        self.env_config.no_of_cars = no_of_cars
        self.load_qvalues(inputfile_number)
        skip = 0
        for episode in range(0, episodes):
            self.epsilon = give_action_choose_probability(episodes, episode, self.probability_model)
            env.reset()
            env.initialize_cars(no_of_cars)
            env.select_reward_function('q_reward')
            """
            Defining initial sources and destinations for cars.
            We are also storing defined source into 'given_nodes' list.
            Which will be used later in exploitation part.
            """
            env.initialize(destination_index_array, input_car_states, destination_choose_arrays)
            # print('episode', episode, self.epsilon)
            self.training_done = 0
            counts_in_one_episode = 0
            self.episode_array = []
            deadlock_count = 0  # TODO: EXTRA
            deadlock_detected = 0  # TODO: EXTRA
            while not self.training_done:
                if deadlock_detected == 1:
                    # print("deadlock detected")
                    break
                counts_in_one_episode += 1
                for num, i in enumerate(env.get_cars()):
                    # num0 = self.car_one_to_one_map_dict[num]
                    if env.check_car_training(num):
                        continue
                    state = env.encodestate(num, 0)
                    action = self.getaction(state, self.epsilon)

                    #     self.update_simulator_display(0)
                    # print ("encoded_state", state, "action_given", action)
                    if action == 'S' and self.epsilon == 0:
                        deadlock_count += 1
                        if deadlock_count == env.no_of_cars * 3:
                            if state not in env.bad_states:
                                # print("deadlock_detected")
                                # env.update_simulator_display(3000)
                                save_deadlock_state(state, env.destination_box_list, env.environment_bad_states_file,
                                                    state.count('z') + 1, max_cars)
                                deadlock_detected = 1
                                break
                            else:
                                skip = 1
                    else:
                        deadlock_count = 0
                    self.episode_array.append(action)
                    # print(action, "Action from detect deadlock mehod")
                    next_state = env.get_successor_state(deepcopy(state), action)

                    if skip:  # Added this for non-solvable states
                        return False
                    if self.epsilon <= 0:
                        pass
                    self.update(state, action, next_state,
                                env.reward_function(deepcopy(state), deepcopy(next_state), num, action))
                    next_state_array = env.state_to_array(next_state)
                    env.update_car(num, deepcopy(next_state_array[0][0]), deepcopy(next_state_array[0][1]), 0)
                    if self.epsilon <= 0:
                        pass
                        # env.update_simulator_display(100)
                self.training_done = env.is_game_ended()
            # print('counts_in_one_episode', counts_in_one_episode)
            if deadlock_detected == 1:
                return episode

        self.save_qvalues(outputfile_number)
        return episode

    def validate_given_state(self, episodes, inputfile_number, outputfile_number, input_car_states=None, no_of_cars=1,
                             destination_index_array=[0, 0, 0, 0], destination_choose_arrays=[0, 0, 0], waitkey=0):
        """
        This function validates the state by using the simulator and running the policy that was learnt during training
        to visualize the results
        :param episodes:
        :param inputfile_number:
        :param outputfile_number:
        :param input_car_states:
        :param no_of_cars:
        :param destination_index_array:
        :param destination_choose_arrays:
        :param waitkey:
        :return:
        """
        self.env_config.no_of_cars = no_of_cars
        self.load_qvalues(inputfile_number)
        for i in range(0, episodes):
            self.validation_done = 0
            self.epsilon = 0
            env.reset()
            env.initialize_cars(no_of_cars)
            """
            Defining initial sources and destinations for cars.
            """
            env.initialize(destination_index_array, input_car_states, destination_choose_arrays)
            print(episodes)
            counts_in_one_episode = 0
            self.episode_array = []
            self.validation_done = 0
            self.epsilon = 0
            while not self.validation_done:
                counts_in_one_episode += 1
                num = env.update_simulator_display(waitkey) - 48
                print("number passes by key press", num)
                if num >= env.no_of_cars:
                    print("in valid car id")
                    continue
                # print ("car0", self.cars[0].source_node)
                # print ("car1", self.cars[1].source_node)
                if env.check_car_training(num):
                    continue
                # print (i.source_node, "source_node")
                state = env.encodestate(num, 1)
                action = self.getaction(state, 0)
                print("for state,", state, "action provided", action)
                next_state = env.get_successor_state(deepcopy(state), action)
                next_state_array = env.state_to_array(next_state)
                print(next_state_array, "next_state_array")
                env.update_car(num, deepcopy(next_state_array[0][0]), deepcopy(next_state_array[0][1]))
                env.update_simulator_display(waitkey)
                self.validation_done = env.is_game_ended()
                if self.validation_done:
                    break

    def convert_textfile_in_ascending_order(self, inputfile_number, outputfile_number):
        """
        Used for converting the states in a text file into ascending order
        :param inputfile_number:
        :param outputfile_number:
        :return:
        """
        self.load_qvalues(inputfile_number)
        self.qvals_new = Counter()
        for state, action in self.qvals.keys():
            print(state, "state")
            array = env.state_to_array(state)
            ascended_array = env.ascend_array(array)
            ascended_state = env.array_to_state(ascended_array)
            print(ascended_state, "ascended_state")
            self.qvals_new[(ascended_state, action)] = self.qvals[(state, action)]
        self.save_qvalues(outputfile_number)

    def convert_textfile(self, input_textfile, output_textfile, block_number):
        """
        This method will be used in converting text file for MSV.
        :param block_number:
        :return:
        """
        input_textfile = './src/data/qvalue_files/' + input_textfile + '.txt'

        output_textfile = './src/data/qvalue_files/' + output_textfile + '.txt'
        """
        block1
        """
        box_conversion_map_1 = {
            'D1': 'A1',
            'D2': 'A2',
            'D3': 'A3',
            'D4': 'A4',
            'D5': 'A5',
            'D6': 'A6',
            'D7': 'A7',
            'D8': 'A8',
            'D9': 'A9'
        }
        """
        block2
        """
        box_conversion_map_2 = {
            'D1': 'A7',
            'D2': 'A8',
            'D3': 'A9',
            'D4': 'A10',
            'D5': 'A11',
            'D6': 'A12',
            'D7': 'A13',
            'D8': 'A14',
            'D9': 'A15'
        }
        """
        block3
        """
        box_conversion_map_3 = {
            'U1': 'B7',
            'U2': 'B8',
            'U3': 'B9',
            'U4': 'B10',
            'U5': 'B11',
            'U6': 'B12',
            'U7': 'B13',
            'U8': 'B14',
            'U9': 'B15'
        }
        """
        block0
        """
        box_conversion_map_0 = {
            'U1': 'B1',
            'U2': 'B2',
            'U3': 'B3',
            'U4': 'B4',
            'U5': 'B5',
            'U6': 'B6',
            'U7': 'B7',
            'U8': 'B8',
            'U9': 'B9'
        }
        box_maps_dict = {0: box_conversion_map_0, 1: box_conversion_map_1, 2: box_conversion_map_2,
                         3: box_conversion_map_3}
        box_conversion_map = box_maps_dict[block_number]

        f_read = open(input_textfile, 'r')
        f_write = open(output_textfile, 'w+')

        for i in f_read.read().split('\n'):
            print(i, "before")
            for key in box_conversion_map.keys():
                i = i.replace(key + 'x', box_conversion_map[key] + 'x')
                i = i.replace(key + 'z', box_conversion_map[key] + 'z')
                i = i.replace(key + 'N', box_conversion_map[key] + 'N')
                i = i.replace(key + 'E', box_conversion_map[key] + 'E')
                i = i.replace(key + 'W', box_conversion_map[key] + 'W')
                i = i.replace(key + 'S', box_conversion_map[key] + 'S')
                i = i.replace(key + '|', box_conversion_map[key] + '|')
            print(i, "after")
            f_write.write(i + '\n')
        f_read.close()
        f_write.close()

    def reverse_convert_textfile(self, input_textfile, output_textfile, block_number):
        """
        This method will be used in converting text file for MSV.
        :param block_number:
        :return:
        """
        input_textfile = './src/data/qvalue_files/' + input_textfile + '.txt'

        output_textfile = './src/data/qvalue_files/' + output_textfile + '.txt'
        """
        block1
        """
        box_conversion_map_1 = {'A1': 'D1', 'A3': 'D3', 'A2': 'D2', 'A5': 'D5', 'A4': 'D4', 'A7': 'D7', 'A6': 'D6',
                                'A9': 'D9', 'A8': 'D8'}

        """
        block2
        """
        box_conversion_map_2 = {'A15': 'D9', 'A14': 'D8', 'A11': 'D5', 'A10': 'D4', 'A13': 'D7', 'A12': 'D6',
                                'A7': 'D1', 'A9': 'D3', 'A8': 'D2'}

        """
        block3
        """
        box_conversion_map_3 = {'B7': 'U1', 'B14': 'U8', 'B15': 'U9', 'B12': 'U6', 'B13': 'U7', 'B10': 'U4',
                                'B11': 'U5', 'B8': 'U2', 'B9': 'U3'}

        """
        block0
        """
        box_conversion_map_0 = {'B4': 'U4', 'B5': 'U5', 'B6': 'U6', 'B7': 'U7', 'B1': 'U1', 'B2': 'U2', 'B3': 'U3',
                                'B8': 'U8', 'B9': 'U9'}

        box_maps_dict = {0: box_conversion_map_0, 1: box_conversion_map_1, 2: box_conversion_map_2,
                         3: box_conversion_map_3}
        box_conversion_map = box_maps_dict[block_number]

        f_read = open(input_textfile, 'r')
        f_write = open(output_textfile, 'w+')

        for i in f_read.read().split('\n'):
            print(i, "before")
            for key in box_conversion_map.keys():
                i = i.replace(key + 'x', box_conversion_map[key] + 'x')
                i = i.replace(key + 'z', box_conversion_map[key] + 'z')
                i = i.replace(key + 'N', box_conversion_map[key] + 'N')
                i = i.replace(key + 'E', box_conversion_map[key] + 'E')
                i = i.replace(key + 'W', box_conversion_map[key] + 'W')
                i = i.replace(key + 'S', box_conversion_map[key] + 'S')
                i = i.replace(key + '|', box_conversion_map[key] + '|')
            print(i, "after")
            f_write.write(i + '\n')
        f_read.close()
        f_write.close()


class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0: return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = self.items()
        compare = lambda x, y: sign(y[1] - x[1])
        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in y.items():
            self[key] += value

    def __add__(self, y):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__(self, y):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend
