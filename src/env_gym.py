from src.configuration import *


class Environment:
    def __init__(self, env_conf):
        self.action_mapping = {
            'S': 0,
            'L': 1,
            'R': 2,
            'F': 3,
            'B': 4,
            'T': 5
        }
        self.reverse_action_mapping = {
            0: 'S',
            1: 'L',
            2: 'R',
            3: 'F',
            4: 'B',
            5: 'T'
        }
        # Configuration object used to unpack values related to training
        self.env_conf = env_conf
        self.box_node_list = self.env_conf.box_node_list
        self.node_action_list = self.env_conf.action_dictionary
        # List containing boxes that are not accessible in the environment for a particular layout
        self.no_box_list = self.env_conf.no_box_list
        # List containing nodes that are not accessible in the environment for a particular layout
        self.no_node_list = self.env_conf.no_node_list
        self.no_of_cars = self.env_conf.no_of_cars

        self.no_of_stations = self.env_conf.no_of_stations

        # As soon as a destination is reached, that particular box is blocked for other cars for a limited count/steps.
        # That count is stored inside the box_count_dict. More in update_car()
        self.box_count_dict = {}

        self.block_destination_count = 0

        # src_dest_length_dict stores the value between a given (source, destination) pair
        self.src_dest_length_dict = {}

        self.destination_nodes_list = self.env_conf.destination_nodes_list
        self.destination_box_list = self.env_conf.station_box_list
        self.selective_blocked_nodes_dictionary = self.env_conf.selective_blocked_nodes_dictionary
        self.selective_blocked_boxes_dictionary = self.env_conf.selective_blocked_boxes_dictionary
        self.blocked_nodes_list = deepcopy(self.env_conf.blocked_nodes_list)
        self.stations = []

        # Holds the reward function object which can be set when running the algorithm
        self.reward_function = None

        # This list consists of objects of the Car class and contains information useful for navigation such as its
        # current source, destination, a flag that tells us whether that particular car has completed its training or
        # not etc. Defined in configuration/configuration.py
        self.cars = []

        self.training_complete_destination_index = self.env_conf.training_complete_destination_index

        # Holds the display object which uses open-cv for rendering a window of the environment. Used for displaying
        # the positions of the cars currently used for training. Defined in configuration/configuration.py
        self.disp = Display(env_conf)

        for i in range(0, self.no_of_stations):
            self.stations.append(Station())
            self.stations[i].station_index = i
            self.stations[i].station_box = self.env_conf.stations_info[i][0]
            self.stations[i].station_node = self.env_conf.stations_info[i][1]
            self.stations[i].processing_count = self.env_conf.stations_info[i][2]

        if not (hasattr(self, 'give_bad_state_reward')):
            try:
                self.environment_bad_states_file = self.env_conf.environment_bad_states_file
                self.bad_states = []
                self.give_bad_state_reward = True
                with open(self.environment_bad_states_file, 'r') as f:
                    for i in f.read().split('\n'):
                        if i == '':
                            continue
                        else:
                            self.bad_states.append(i)
            except:
                self.give_bad_state_reward = False

    def step(self, state, action, num):
        """
        This function is responsible for stepping through the environment one action at a time. Similar to the step in
        gym environments. Updates the current source and destination of the car after an action.

        :param state: state of the environment
        :param action: action to be taken
        :param num: index of the car present in training
        :return: state, action, next_state, reward, next_state_array
        """
        next_state = self.get_successor_state(deepcopy(state), action)
        reward = self.reward_function(deepcopy(state), deepcopy(next_state), num, action)
        next_state_array = self.state_to_array(next_state)
        self.update_car(num, deepcopy(next_state_array[0][0]), deepcopy(next_state_array[0][1]), 0)
        return state, action, next_state, reward, next_state_array

    def select_reward_function(self, name):
        """
        Assign the reward function of your choice by defining it in the
        environment class

        :param name: Reward function name
        :return: variable containing the reward function object to be used during training
        """
        if name == 'q_reward':
            self.reward_function = self.q_reward
        elif name == 'dqn_reward':
            self.reward_function = self.dqn_reward

    def get_legal_actions(self, node):
        """
        This function is responsible for calculating the total number of actions for a given node. These actions are
        unfiltered. DYY/UYY are values used for representing nodes that don't exist/are unreachable and are skipped.

        :param node: Node of a car
        :return: list of available actions
        """
        actions = self.node_action_list[node]
        unfiltered_actions = []
        for i in range(len(actions)):
            if actions[i] == 'DYY' or actions[i] == 'UYY':
                continue
            if actions[i] in self.no_node_list:
                continue
            unfiltered_actions.append(self.reverse_action_mapping[i])
        return deepcopy(unfiltered_actions)

    def get_successor_node(self, node, action):
        """
        gives successor node of a state based on action
        :param node: node of a car
        :param action: action to be taken
        :return: successor node
        """
        # print ("node", node, "action", action)
        index = self.action_mapping[action]
        return self.node_action_list[node][index]

    def check_selective_blocked_nodes_condition(self, state, action):
        """
        This function is responsible for avoiding deadlocks up to a limit in the system.
        1) The destination box and successor box of the primary car is calculated
        2) We check three factors
            2a) the primary car has a destination present in the selective_block_node keys
            2b) the primary car has a successor box in the value of that particular selective_block_node key
        `   2c) If any secondary car is already present in one of the values of that particular selective_block_node key

        :param state: state of the environment
        :param action: action taken
        :return: flag indicating the particular action will result in a future crash or not
        """
        if action == 'S' or action == 'L' or action == 'R' or action == 'T':
            return 0
        if action == 'B':
            return 1
        primary_car_source_node = state[0][0]
        primary_car_successor_node = self.get_successor_node(primary_car_source_node, action)
        primary_car_destination_node = state[0][1]
        primary_car_source_box = str(primary_car_source_node[0]) + str(
            int(re.search(r'\d+', primary_car_source_node).group()))
        """ 
        1
        """
        primary_car_destination_box = str(primary_car_destination_node[0]) + str(
            int(re.search(r'\d+', primary_car_destination_node).group()))
        primary_car_successor_box = str(primary_car_successor_node[0]) + str(
            int(re.search(r'\d+', primary_car_successor_node).group()))

        """
        2
        """
        for i in self.selective_blocked_boxes_dictionary.keys():
            if i != primary_car_destination_box:
                continue
            if primary_car_successor_box not in self.selective_blocked_boxes_dictionary[i]:
                continue
            for num, j in enumerate(state):
                if num == 0:
                    continue
                if j[0] in self.selective_blocked_boxes_dictionary[i]:
                    return 1
        return 0

    def getlegalactions_filtered(self, state):
        """
        gives legal action with filtering. Method does following steps.
        1) Extracting information of primary car.(Primary car is the car from all the cars who asks for action.) like
           source node, destination node.
        For all actions check the following.
        2) Check if selective blocked nodes condition is violated or not. if it violates car will crash, doesn't append
         action as crashed_flag will be high.
        3) Check if successor node is in blocked nodes list or not and action is other than Stop. If true then action is not appended.


        :param state: state of the environment
        :return: list of actions
        """

        primary_car_source_node = state[0][0]

        temp_actions = self.get_legal_actions(primary_car_source_node)
        actions = []
        for i in temp_actions:
            successor_node = self.get_successor_node(primary_car_source_node, i)
            crashed_flag = self.check_selective_blocked_nodes_condition(state, i)

            for num, j in enumerate(state):
                if num == 0:
                    continue
                successor_box = int(re.search(r'\d+', successor_node).group())
                box_tobe_compared = int(re.search(r'\d+', j[0]).group())
                if successor_box == box_tobe_compared:
                    crashed_flag = 1
            if successor_node in self.blocked_nodes_list and i != 'S':
                continue
            if crashed_flag:
                continue
            else:
                actions.append(i)
        return actions

    def state_to_array(self, state):
        '''
        This function mapps a state of qlearning into a cars source destinations array
        :param state:
        :return: array
        '''
        array = []
        for i in state.split('z'):
            array.append(i.split('x'))
        return array

    def array_to_state(self, array):
        '''
        This function mapps a cars source destinations array into a qlearning state
        :param array:
        :return: states
        '''
        state = ''
        for i in array:
            for num, j in enumerate(i):
                state += (j)
                if num == 0:
                    state += ('x')
                if num == 1:
                    state += ('z')
        return state[:-1]

    def getlegalactions(self, state):
        return self.getlegalactions_filtered(state)

    def get_successor_state(self, state, action):
        """
        gives successor state based on current state and action to be taken
        :param state: current state
        :param action: action to be taken
        :return: state
        """
        array = self.state_to_array(state)
        array[0][0] = deepcopy(self.get_successor_node(array[0][0], action))
        return self.array_to_state(array)

    def ascend_array(self, array):
        temp_array = deepcopy(array)
        temp_array.remove(temp_array[0])
        index_array = []
        for i in temp_array:
            source = i[0]
            index_array.append((int(re.search(r'\d+', source).group()), i))
        index_array.sort()
        temp_array = []
        for i, j in index_array:
            temp_array.append(j)
        temp_array.insert(0, array[0])
        return temp_array

    def encodestate(self, car_id, deadlock_id):
        """
        This function encodes the information (source and destination) in states for cars. Our learning
        model can understand this state encoding
        the car for which this function is called is encoded as (Floor Box_number Node) ;Example: D2W
        While for other cars only (Floor Box_number) is encoded; Example : D2
        The source and destination for a particular file is separated with character 'x' between them.
        The source destination pair of two or more cars is separated by z

        1) The information of the primary car is extracted before hand and stored inside cars_array
        2) For the secondary cars, we only add the source and destination box and append them to the cars array
        :param self: object of initialization
        :param car_id: id of the car
        :return state
        """

        """1"""
        primary_car_source_node = self.cars[car_id].source_node
        primary_car_destination_node = self.destination_nodes_list[self.cars[car_id].destination_index][
            self.cars[car_id].destination_choose_array[self.cars[car_id].destination_index]]
        cars_array = [[primary_car_source_node, primary_car_destination_node]]

        """2"""
        for num, i in enumerate(self.cars):
            if num == car_id:
                continue
            if deadlock_id == 1:
                if i.training_done:
                    i.training_done_count -= 1
                    if i.training_done_count < 0:
                        continue
                    else:
                        source_node = self.destination_nodes_list[self.training_complete_destination_index][
                            i.destination_choose_array[self.training_complete_destination_index]]
                        source_box = str(source_node[0]) + str(int(re.search(r'\d+', source_node).group()))
                        array = [source_box, source_box]
                        cars_array.append(array)
                        continue
            if deadlock_id == 0:
                if i.training_done:
                    continue
            source_node = i.source_node
            source_box = str(source_node[0]) + str(int(re.search(r'\d+', source_node).group()))
            destination_node = self.destination_nodes_list[i.destination_index][
                i.destination_choose_array[i.destination_index]]
            destination_box = str(destination_node[0]) + str(int(re.search(r'\d+', destination_node).group()))
            array = [source_box, destination_box]
            cars_array.append(array)
        return self.array_to_state(self.ascend_array(cars_array))

    def initialize(self, destination_index_array, input_car_states, destination_choose_arrays):
        """
        This function takes in the index array, car state and choose array and initializes the initial positions of the
        all the cars based on these values. The positions are decided randomly.
        :param input_car_states: The car state that needs to be trained.
        :param destination_index_array: The array that decides which destinations the car will take based on available destinations.
        :param destination_choose_arrays:
        :return:
        """
        if input_car_states is not None:
            self.given_nodes = []
            for num, i in enumerate(self.cars):
                # print(num, i)
                # self.terminal_logger.info('Car number ' + str(num))
                nodes = deepcopy(self.box_node_list[input_car_states[num]])
                random.shuffle(nodes)

                self.cars[num].source_node = nodes[0]
                self.given_nodes.append(nodes[0])

                self.cars[num].destination_index = destination_index_array[num]
                self.cars[num].destination_choose_array = destination_choose_arrays[num]

    def reinitialize(self, destination_index_array, input_car_states, destination_choose_arrays):
        """
        This function takes in the index array, car state and choose array and re-initializes the positions of the car
        using the actions that were stored while running an episode in the experiment.
        :param input_car_states: The car state that needs to be trained.
        :param destination_index_array: The array that decides which destinations the car will take based on available destinations.
        :param destination_choose_arrays:
        :return:
        """
        if input_car_states is not None:
            for num, i in enumerate(self.cars):
                self.cars[num].source_node = self.given_nodes[num]
                self.cars[num].destination_index = destination_index_array[num]
                self.cars[num].destination_choose_array = destination_choose_arrays[num]

    def block_box(self, boxes):
        for box in boxes:
            self.block_nodes(self.box_node_list[box])

    def unblock_box(self, boxes):
        for box in boxes:
            self.unblock_nodes(self.box_node_list[box])

    def block_nodes(self, nodes):
        self.blocked_nodes_list.extend(nodes)
        self.blocked_nodes_list = list(set(self.blocked_nodes_list))

    def unblock_nodes(self, nodes):
        for i in nodes:
            if i in self.blocked_nodes_list:
                self.blocked_nodes_list.remove(i)

    def update_car(self, car_id, source_node, destination_node, block_destination_nodes=0):
        """
        This function updates the source and destination for the cars in the system. This operation is performed only
        for the primary car.
        1) If block_destination nodes is 1 and a box is available in the box_count keys then,
            1a) The box_count for that node is decreased by 1
            1b) If the box_count is 0, then unblock that box and delete entry for that box from the box_count dictionary

        2) Check whether the source and destination of the car are same
            2a) If True, then update the destination of that particular car
            2b) If the car has reached its final destination, then assign that car a position from the no_box_list
        :param car_id: id of the car
        :param source_node: source_node of the car
        :param destination_node: destination_node of the car
        :return state
        """
        """1"""

        block_destination_nodes = 0
        if block_destination_nodes == 1 and len(self.box_count_dict.keys()) != 0:
            for i in list(self.box_count_dict.keys()):
                self.box_count_dict[i] -= 1
                # print(self.box_count_dict[i])
                # input()
                if self.box_count_dict[i] == 0:
                    self.unblock_box([i])
                    del self.box_count_dict[i]

        """2"""

        self.cars[car_id].source_node = source_node
        self.cars[car_id].destination_node = destination_node

        """2a"""
        if self.cars[car_id].source_node == self.destination_nodes_list[self.cars[car_id].destination_index][self.cars[car_id].destination_choose_array[self.cars[car_id].destination_index]]:

            """
            if we want to block destination nodes.
            this if condition will be executed.
            """
            if block_destination_nodes == 1:
                dest_node = self.destination_nodes_list[self.cars[car_id].destination_index][
                    self.cars[car_id].destination_choose_array[self.cars[car_id].destination_index]]
                dest_box = str(dest_node[0]) + str(int(re.search(r'\d+', dest_node).group()))

                self.block_box([dest_box])
                self.block_destination_count = 5
                self.box_count_dict[dest_box] = self.block_destination_count

            """2b"""

            if self.cars[car_id].destination_index == self.training_complete_destination_index:
                self.cars[car_id].training_done = 1
                self.cars[car_id].source_node = self.no_box_list[0] + 'W'
            self.cars[car_id].destination_index += 1
            self.cars[car_id].destination_index %= len(self.destination_nodes_list)

    def breadth_first_search(self, cars_source_destinations_pairs):
        """
        This function performs the breadth first search calculation between the source and the destination of a car and
        stores the actions taken to reach the goal. It also keeps track of the nodes that were visited during the
        calculation of the optimal path.
        :param cars_source_destinations_pairs
        :return: list of actions taken, list of visited nodes
        """
        source_node = deepcopy(cars_source_destinations_pairs[0][0])
        destination_node = deepcopy(cars_source_destinations_pairs[0][1])

        bfs_fringe = Queue()
        bfs_fringe.push([source_node, [], []])
        bfs_closedset = {source_node}

        while not bfs_fringe.isEmpty():

            node = bfs_fringe.pop()
            node_in_search = deepcopy(node[0])
            legal_actions = self.get_legal_actions(node_in_search)

            for action in legal_actions:
                if action == 'B':
                    continue
                actions_of_searchnode = deepcopy(node[1])
                visited_boxes = deepcopy(node[2])
                item = deepcopy(self.get_successor_node(node_in_search, action))
                # first action is stop so code also checks whether starting and destination nodes are same.
                if item == destination_node:
                    actions_of_searchnode.append(action)
                    visited_boxes.append(item[:-1])
                    return actions_of_searchnode, visited_boxes
                if item in bfs_closedset:
                    continue
                actions_of_searchnode.append(action)
                visited_boxes.append(item[:-1])

                bfs_fringe.push([item, actions_of_searchnode, visited_boxes])
                bfs_closedset |= {item}

        return ['S'], []

    def dqn_reward(self, previous_state, state, car_id, action='S'):
        """
        This is the reward function designed for dqn
        1) Uses Breadth First Search for calculating distance ( in steps )
           between the previous state and next state and uses the difference to calculate whether the reward should be
           positive or negative
        2) An extra reward is added for handling deadlock states. It uses the optimal paths of the primary and
           remaining cars after the action is taken by the primary car and provides an extremely negative reward
           if the condition is satisfied
        3) Finally the rewards are scaled to between [-1, 1] by dividing the total reward by 100
        :param previous_state, state, car_id:
        :return: reward
        """
        # if self.give_bad_state_reward:
        #     temp_check_previous_state = deepcopy(previous_state)
        #     temp_check_state = deepcopy(state)
        #     for i in ['N', 'E', 'W', 'S']:
        #         temp_check_previous_state = temp_check_previous_state.replace(i, '')
        #         temp_check_state = temp_check_state.replace(i, '')
        #     if temp_check_state in self.bad_states:
        #         return -10000.0
        #     if temp_check_previous_state in self.bad_states:
        #         return -10000.0
        state_array = self.state_to_array(state)
        previous_state_array = self.state_to_array(previous_state)
        previous_source = previous_state_array[0][0]
        source = state_array[0][0]
        destination = state_array[0][1]
        current_pair = (source, destination)
        previous_pair = (previous_source, destination)
        if current_pair in self.src_dest_length_dict.keys():
            current_steps = self.src_dest_length_dict[current_pair][0]
        else:
            actions, visited_nodes = self.breadth_first_search([[source, destination]])
            current_steps = len(actions)
            self.src_dest_length_dict[current_pair] = (current_steps, visited_nodes)
        if previous_pair in self.src_dest_length_dict.keys():
            previous_steps = self.src_dest_length_dict[previous_pair][0]
        else:
            actions, visited_nodes = self.breadth_first_search([[previous_source, destination]])
            previous_steps = len(actions)
            self.src_dest_length_dict[previous_pair] = (previous_steps, visited_nodes)

        """
        1. Here the previous steps and current steps have been calculated using the previous state and current state.
           Their difference is used below.
        """

        step_difference = previous_steps - current_steps
        reward = -10
        reward += 50 * step_difference
        if state_array[0][0] == state_array[0][1]:
            reward += 110

        """
        2. The below statement checks if the primary car is crashing into some other car or not and provides a negative
           reward for the same
        """

        for num, i in enumerate(self.cars):
            if num == car_id:
                continue
            state_box = int(re.search(r'\d+', state_array[0][0]).group())
            car_box = int(re.search(r'\d+', i.source_node).group())
            if (state_box == car_box) and (state_array[0][0][0] == i.source_node[0]):
                # print("car_id",car_id,"state_box",state_box,"car_box",car_box,"car_num",num,"i.source_node",i.source_node,"array00",array[0][0])
                reward -= 100  # in case of clash; next state of car is source of different car.

        return reward/100

    def q_reward(self, previous_state, state, car_id, action='S'):
        """
        This is the reward function designed for q-learning
        1) Uses Breadth First Search for calculating distance ( in steps )
           between the previous state and next state and uses the difference to calculate whether the reward should be
           positive or negative
        2) An extra reward is added for handling deadlock states. It uses the optimal paths of the primary and
           remaining cars after the action is taken by the primary car and provides an extremely negative reward
           if the condition is satisfied
        :param previous_state, state, car_id:
        :return: reward
        """

        if self.give_bad_state_reward:
            temp_check_previous_state = deepcopy(previous_state)
            temp_check_state = deepcopy(state)
            for i in ['N', 'E', 'W', 'S']:
                temp_check_previous_state = temp_check_previous_state.replace(i, '')
                temp_check_state = temp_check_state.replace(i, '')
            if temp_check_state in self.bad_states:
                return -10000.0
            if temp_check_previous_state in self.bad_states:
                return 0
        state_array = self.state_to_array(state)
        previous_state_array = self.state_to_array(previous_state)
        previous_source = previous_state_array[0][0]
        source = state_array[0][0]
        destination = state_array[0][1]
        current_pair = (source, destination)
        previous_pair = (previous_source, destination)
        if current_pair in self.src_dest_length_dict.keys():
            current_steps = self.src_dest_length_dict[current_pair][0]
        else:
            actions, visited_nodes = self.breadth_first_search([[source, destination]])
            current_steps = len(actions)
            self.src_dest_length_dict[current_pair] = (current_steps, visited_nodes)
        if previous_pair in self.src_dest_length_dict.keys():
            previous_steps = self.src_dest_length_dict[previous_pair][0]
        else:
            actions, visited_nodes = self.breadth_first_search([[previous_source, destination]])
            previous_steps = len(actions)
            self.src_dest_length_dict[previous_pair] = (previous_steps, visited_nodes)

        """
        1. Here the previous steps and current steps have been calculated using the previous state and current state.
           Their difference is used below.
        """

        step_difference = previous_steps - current_steps
        reward = -10
        reward += 50 * step_difference
        if state_array[0][0] == state_array[0][1]:
            reward += 10010

        """
        2. The below statement checks if the primary car is crashing into some other car or not and provides a negative
           reward for the same
        """
        for num, i in enumerate(self.cars):
            if num == car_id:
                continue
            state_box = int(re.search(r'\d+', state_array[0][0]).group())
            car_box = int(re.search(r'\d+', i.source_node).group())
            if (state_box == car_box) and (state_array[0][0][0] == i.source_node[0]):
                reward -= 100000  # in case of clash; next state of car is source of different car.
        return reward

    def is_game_ended(self):
        """
        This function counts the total cars whose training has ended and tells us whether the episode has finished or
        not.
        :param :
        :return: True or False
        """
        count = 0
        for i in self.cars:
            if i.training_done:
                count += 1
        if count == self.no_of_cars:
            self.blocked_nodes_list = []
            self.box_count_dict = {}
            return True
        else:
            return False

    def get_cars(self):
        return self.cars

    def reset(self):
        self.__init__(self.env_conf)

    def check_car_training(self, num):
        """
        This function checks whether a particular car has finished training or not
        not.
        :param num: car_id
        :return: True or False
        """
        return self.cars[num].training_done

    def initialize_cars(self, no_of_cars):
        """
        Here Car objects are initialized with their initial source and destinations along with other properties and
        appended into the cars list.
        :param no_of_cars: Number of cars required in the system
        """
        self.env_conf.no_of_cars = no_of_cars
        self.no_of_cars = no_of_cars
        for i in range(0, self.no_of_cars):
            self.cars.append(Car(self.env_conf))
            self.cars[i].car_id = i
            self.cars[i].source_node = self.env_conf.cars_info[i][0]
            self.cars[i].temp_source_node = self.cars[i].source_node
            self.cars[i].color = self.env_conf.cars_info[i][1]
            self.cars[i].move_counter_maxvalue = self.env_conf.cars_info[i][2]
            self.cars[i].training_done = 0
            self.cars[i].destination_index = self.env_conf.cars_info[i][3]
            self.cars[i].destination_choose_array = self.env_conf.cars_info[i][4]

    def update_simulator_display(self, waitkey):
        """
        This function is used for rendering the open-cv window on the screen. It uses the source nodes of the cars to
        display them.
        :param waitkey: Integer required for holding the open-cv window for appropriate number of seconds
        """
        cars_location = []
        for i in self.cars:
            cars_location.append((i.source_node, self.get_successor_node(i.source_node, 'L'),
                                 self.get_successor_node(i.source_node, 'R')))
        station_state = []
        for i in self.stations:
            station_state.append(i.is_free)
        return self.disp.update(cars_location, station_state, waitkey)


# The following code was taken from the UC-Berkeley CS188 Pacman Course
# http: // inst.eecs.berkeley.edu / ~cs188 / pacman / pacman.html

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0


