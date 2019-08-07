import numpy as np
from copy import deepcopy
import random
from utils.utils import *
import re
import tensorflow as tf
from environment_creater_3d import Environment
from collections import deque
import datetime
from logging_utils import setup_logger
import logging
import os
from keras.models import load_model
from settings import MODEL_DIRECTORY
from ordered_set import OrderedSet


# random.seed(5)


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred, delta=0.1)


# def weight_initializer(shape):
#     initializer = tf.contrib.layers.xavier_initializer_conv2d()
#     variable = tf.Variable(initializer(shape=shape))
#     return variable
#
#
# def bias_initializer(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# class NoisyNetDQN:
#     def __init__(self, inp, _y, lr, scope='default', use_ewc=0):
#         self.lr = lr
#         self.use_ewc = use_ewc
#     # with tf.variable_scope(scope):
#         self.input_layer = inp
#         self.input_dim = int(self.input_layer.get_shape()[1])
#         self.output_dim = int(_y.get_shape()[1])
#         self.w1 = weight_initializer([self.input_dim, 24])
#         self.b1 = bias_initializer([24])
#         self.w2 = weight_initializer([24, 24])
#         self.b2 = bias_initializer([24])
#         self.w3 = weight_initializer([24, 24])
#         self.b3 = bias_initializer([24])
#         self.w4 = weight_initializer([24, 24])
#         self.b4 = bias_initializer([24])
#         self.w5 = weight_initializer([24, 24])
#         self.b5 = bias_initializer([24])
#         self.w6 = weight_initializer([24, self.output_dim])
#         self.b6 = bias_initializer([self.output_dim])
#         self.var_list = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5,
#                          self.b5, self.w6, self.b6]
#         self.fisher = []
#
#         self.h1 = tf.nn.relu(tf.matmul(self.input_layer, self.w1) + self.b1)
#         self.h1 = tf.keras.layers.BatchNormalization()(self.h1)
#         self.h2 = tf.nn.relu(tf.matmul(self.h1, self.w2) + self.b2)
#         self.h2 = tf.keras.layers.BatchNormalization()(self.h2)
#         self.h3 = tf.nn.relu(tf.matmul(self.h2, self.w3) + self.b3)
#         self.h3 = tf.keras.layers.BatchNormalization()(self.h3)
#         self.h4 = tf.nn.relu(tf.matmul(self.h3, self.w4) + self.b4)
#         self.h4 = tf.keras.layers.BatchNormalization()(self.h4)
#         self.h5 = tf.nn.relu(tf.matmul(self.h4, self.w5) + self.b5)
#         self.y = tf.nn.softmax(tf.matmul(self.h5, self.w6) + self.b6)
#         self.mean_square_error = tf.losses.mean_squared_error(predictions=self.y, labels=_y)
#         self.train_step = None
#         self.ewc_loss = None
        # self.vanilla_loss()

    # def vanilla_loss(self):
    #     self.ewc_loss = self.mean_square_error
    #     self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.ewc_loss)
#
#     def star(self):
#         """
#         Store the best weights while training
#         :return:
#         """
#         self.latest_weights = []
#         for v in range(len(self.var_list)):
#             self.latest_weights.append(self.var_list[v].eval())
#
#     def restore(self, session):
#         # reassign optimal weights for latest task
#         if hasattr(self, "latest_weights"):
#             for v in range(len(self.var_list)):
#                 session.run(self.var_list[v].assign(self.latest_weights[v]))
#
#     def set_ewc_loss(self, lam=1):
#         """
#         Loss for EWC algorithm
#         :param lam:
#         :return:
#         """
#         self.ewc_loss = self.mean_square_error
#         if len(self.fisher) != 0 and self.use_ewc:
#             for v in range(len(self.var_list)):
#                 self.ewc_loss += (lam / 2) * tf.reduce_sum(tf.multiply(self.fisher[v].astype(np.float32),
#                                                                        tf.square(
#                                                                            self.var_list[v] -
#                                                                            self.latest_weights[
#                                                                                v])))
#
#             self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.ewc_loss)
#
#     def compute_fisher(self, sampleset, sess, num_samples=200):
#         self.fisher = []
#         for v in range(len(self.var_list)):
#             self.fisher.append(np.zeros(self.var_list[v].get_shape().as_list()))
#         probs = self.y
#         class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
#         for j in range(num_samples):
#             # select random input image
#             ind = np.random.randint(sampleset.shape[0])
#             # compute first-order derivatives
#             derivatives = sess.run(tf.gradients(tf.log(probs[0, class_ind]), self.var_list),
#                                    feed_dict={self.input_layer: sampleset[ind:ind + 1]})
#             # square the derivatives and add to total
#             for v in range(len(self.fisher)):
#                 self.fisher[v] += np.square(derivatives[v])
#                 self.fisher[v] /= num_samples
#         self.set_ewc_loss()


class RL:
    def __init__(self, input_dim, output_dim, cars, sess, env_config, learning_rate=0.0001, decay_rate=0.00000000001, batch_size=32,
                 representation_type='1', epochs=1, mem_size=5000):
        self.env_config = env_config
        global env
        env = Environment(env_config)
        self.set_finished_box = ""
        # self.x = tf.placeholder(tf.float32, shape=[None, 8 * cars * 2])
        # self.y = tf.placeholder(tf.float32, shape=[None, 5])
        self.sess = sess
        # self.sess_target_model = sess_target_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.annealed_batch_size = batch_size
        self.total_cars = cars
        self.representation_type = representation_type
        self.epochs = epochs
        self.decay_rate = decay_rate
        self.load_model_flag = (0, None)
        self.unusual_sample_factor = 0.9
        self.actions_dict = {
            'S': 0,
            'L': 1,
            'R': 2,
            'F': 3,
            'T': 4,
        }
        self.actions_dict_reverse = {
            0: 'S',
            1: 'L',
            2: 'R',
            3: 'F',
            4: 'T',
        }

        # filepath = "weights.best_via_terminal_region_3.hdf5"
        filepath = "models/ignore.h5"
        self.mem_size = mem_size
        self.memory = deque(maxlen=self.mem_size)
        self.episode_memory = deque(maxlen=self.mem_size)
        self.special_memory = deque(maxlen=self.mem_size)
        self.initialize_logger()
        # self.encode_state_logger = self.create_logger('encode_state_logger', 'INFO')
        # self.update_car_logger = self.create_logger('update_car_logger', 'INFO')
        # self.terminal_logger = self.create_logger('terminal_logger', 'INFO')
        self.qval_logger = self.create_logger('qval_logger', 'INFO')
        self.memory_logger = self.create_logger('memory_logger', 'INFO')
        self.state_logger = self.create_logger('state_logger', 'INFO')
        self.invalid_action_logger = self.create_logger('invalid_action', 'INFO')

        # self.model = self.build_model()
        # self.target_model = self.build_model()
        self.noisy_distribution = 'factorised'
        self.input_tensor = tf.placeholder("float", [None, self.input_dim])
        self.target_input_tensor = tf.placeholder("float", [None, self.input_dim])
        self.output_tensor = tf.placeholder("float", [None, self.output_dim])

        self.optimizer = self.build_model(self.input_tensor, 24, 24, 24, 24)
        self.build_target_model(self.target_input_tensor, 24, 24, 24, 24)
        # self.q_target()
        # self.update_target_net()
        #
        self.saver = tf.train.Saver(max_to_keep=20000)


        # self.save_model()
        # self.restore_model()

        # self.model = NeuralNetwork(self.x, self.y, self.learning_rate, 'model', use_ewc=1)
        # self.target_model = NeuralNetwork(self.x, self.y, self.learning_rate, 'target_model', use_ewc=0)
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # callbacks = EarlyStopping(monitor='val_loss', patience=2)
        # callbacks.set_model(self.model)

        self.alpha = 0.9
        self.discount = 0.15
        self.cars_reached_destination = 0
        self.episode_iterations = 20
        self.probability_model = 3
        self.training_done = 0
        self.discount_model = 1
        self.episode_array = []
        self.epsilon = 0
        self.validation_done = 0
        self.masks = {}
        self.training_complete_destination_index = self.env_config.training_complete_destination_index

        self.sess.run(tf.global_variables_initializer())
        # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)
        # for each in train_vars:
        #     print(each)
        # input()

        # for l in tf.global_variables_initializer():
        #     print(l)
        # input()

    # def build_model(self):
    #     seed = 1
    #     input_layer = Input(shape=(self.input_dim,))
    #     hl = Dense(24, activation="relu", kernel_initializer=he_normal(seed=seed))(input_layer)
    #     hl = BatchNormalization()(hl)
    #     hl = Dense(24, activation="relu", kernel_initializer=he_normal(seed=seed))(hl)
    #     hl = BatchNormalization()(hl)
    #     hl = Dense(24, activation="relu", kernel_initializer=he_normal(seed=seed))(hl)
    #     hl = BatchNormalization()(hl)
    #     hl = Dense(24, activation="relu", kernel_initializer=he_normal(seed=seed))(hl)
    #     hl = BatchNormalization()(hl)
    #     output_layer = Dense(self.output_dim, kernel_initializer=glorot_uniform(seed=seed), activation="linear")(hl)
    #     # output_layer = Dense(self.output_dim, activation="tanh")(hl)
    #     # Also try nothing as an activation function
    #     model = Model(input_layer, output_layer)
    #     model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))
    #     # Can also try huber_loss as a loss function
    #     return model

    def build_model(self, input_tensor, units1, units2, units3, units4):
        """
        This function updates the source and destination for the cars in the system. This operation is performed only
        for the primary car.
        1) If block_destination nodes is 1 and a box is available in the box_count keys then,
            1) The box_count for that node is decreased by 1
            2) If the box_count is 0, then unblock that box and delete entry for that box from the box_count dictionary

        2) Check whether the source and destination of the car are same
            1) If True, then update the destination of that particular car
            2) If the car has reached its final destination, then assign that car a position from the no_box_list
        :param input_tensor: Placeholder for the Input Dimension
        :param units1: size in 1st layer
        :param units2: size in 2nd layer
        :param units3: size in 3rd layer
        :param units4: size in 4th layer
        :return optimizer
        """
        seed = 1
        w_i = tf.keras.initializers.he_normal(seed=seed)
        b_i = tf.constant_initializer(0)
        with tf.variable_scope('model_net', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('l1', reuse=tf.AUTO_REUSE):
                dense1 = noisy_dense(input_tensor, units1, [units1], None, w_i, b_i)
            with tf.variable_scope('l2', reuse=tf.AUTO_REUSE):
                dense2 = noisy_dense(dense1, units2, [units2], None, w_i, b_i)
            with tf.variable_scope('l3', reuse=tf.AUTO_REUSE):
                dense3 = noisy_dense(dense2, units3, [units3], None, w_i, b_i)
            with tf.variable_scope('l4', reuse=tf.AUTO_REUSE):
                dense4 = noisy_dense(dense3, units4, [units4], None, w_i, b_i)
            with tf.variable_scope('final', reuse=tf.AUTO_REUSE):
                w_i = tf.contrib.layers.xavier_initializer(seed=seed)
                self.result = noisy_dense(dense4, self.output_dim, [self.output_dim], None, w_i, b_i, activation=None)

        loss = tf.reduce_mean(tf.losses.huber_loss(self.result, self.output_tensor, delta=0.1))
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def build_target_model(self, target_input_tensor, units1, units2, units3, units4):
        seed = 1
        w_i = tf.keras.initializers.he_normal(seed=seed)
        b_i = tf.constant_initializer(0)
        with tf.variable_scope('target_net', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('l1', reuse=tf.AUTO_REUSE):
                dense1 = noisy_dense(target_input_tensor, units1, [units1], None, w_i, b_i)
            with tf.variable_scope('l2', reuse=tf.AUTO_REUSE):
                dense2 = noisy_dense(dense1, units2, [units2], None, w_i, b_i)
            with tf.variable_scope('l3', reuse=tf.AUTO_REUSE):
                dense3 = noisy_dense(dense2, units3, [units3], None, w_i, b_i)
            with tf.variable_scope('l4', reuse=tf.AUTO_REUSE):
                dense4 = noisy_dense(dense3, units4, [units4], None, w_i, b_i)
            with tf.variable_scope('final', reuse=tf.AUTO_REUSE):
                w_i = tf.contrib.layers.xavier_initializer(seed=seed)
                self.target_result = noisy_dense(dense4, self.output_dim, [self.output_dim], None, w_i, b_i,
                                                 activation=None)

    def update_target_model(self):
        """
        This function updates the target model by copying values from the original model for all the parameters.
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith('model_net')]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith('target_net')]
        e2_params = sorted(e2_params, key=lambda v: v.name)
        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        self.sess.run(update_ops)

    def predict(self, data):
        """
        This function accepts the input data encoding for multiple states and outputs the prediction on each of them.
        :param data: list of data for any number of states. Is a 2d Tensor
        :return predicted data
        """
        predictions = self.result.eval(feed_dict={self.input_tensor: data}, session=self.sess)
        return predictions

    def target_predict(self, data):
        true = self.target_result.eval(feed_dict={self.target_input_tensor: data}, session=self.sess)
        return true

    def save_model(self, path):
        print("Model saved in : ", self.saver.save(self.sess, path))
        
    '''
    def freeze_session(self, sess, keep_var_names=None, output_names=None, clear_devices=True):
        """
        Freezes the state of a session into a pruned computation graph.

        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param sess The TensorFlow session to be frozen.
        @param keep_var_names A list of variable names that should not be frozen,
                              or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices Remove the device directives from the graph for better portability.
        @return The frozen graph definition.
        """
        graph = sess.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

    def save_model(self, path, name):
        # Freeze the graph
        frozen_graph = self.freeze_session(self.sess)
        tf.train.write_graph(frozen_graph, path, name, as_text=False)
     '''

    def load_model(self, session, path):
        session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.saver.restore(session, os.path.join(path, checkpoint_name))
            print(checkpoint_name + ' restored')
            return True
        else:
            return False

    @staticmethod
    def initialize_logger():
        base_path = os.path.join("logs" + str(datetime.datetime.now().strftime("%Y")),
                                 str(datetime.datetime.now().strftime("%m"))
                                 )
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    def create_logger(self, name, level):
        a = setup_logger(name, level=level)
        a = logging.getLogger(name)
        return a

    def give_mask(self, state):
        """
        This function calculates a mask for each state by comparing the total and legal actions available for that state
        :param state:
        :return mask, list of available actions
        """
        total_actions = ['S', 'L', 'R', 'F', 'T']
        valid_actions = env.getlegalactions(env.state_to_array(state))
        mask = np.nonzero(np.in1d(total_actions, valid_actions))[0]
        return mask, valid_actions

    def getaction(self, state):
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

        self.flag = 0
        mask, valid_actions = self.give_mask(state)

        if self.epsilon == 0:
            print(valid_actions)
        action = None
        "*** YOUR CODE HERE ***"
        if len(valid_actions) != 0:
            if flipcoin(self.epsilon):
                action = random.choice(valid_actions)
            else:
                # action = int(np.argmax(
                #     self.sess.run(self.model.y, feed_dict={self.x: self.convert([state], self.total_cars)})[0]))

                qvalues = self.predict(self.convert([state], self.total_cars))[0]
                max_qvalue = np.amax(qvalues[mask])
                action = self.actions_dict_reverse[int(np.argwhere(qvalues == max_qvalue)[0])]
                print(qvalues, action)

        # print(self.model.predict(self.convert([state], self.total_cars))[0])
        if type(action) == str:
            return action, self.actions_dict[action]
        else:
            return self.actions_dict_reverse[action], action

    def convert(self, states, total_cars):
        """
        This function takes in an array consisting of multiple state arrays and converts them into the required notation
        for passing them into the neural network.
        The total_cars have been set to 5 so the neural network representation is being calculated for 5 cars at a
        maximum. -1 is used as a replacement for any remaining cars less than 5
        :param states: The multiple state array
        :param total_cars: total cars used in the system. This has been set to 5 for accommodating 5 cars at max
        :return:
        """
        actions_convert = {
            'S': 0,
            'L': 1,
            'R': 2,
            'F': 3,
            'T': 4,
        }
        direction_convert = {
            'N': 0,
            'W': 1,
            'E': 2,
            'S': 3,
            '-1': -1,
        }
        total_data = []
        for line in states:
            # had to add the string characters (meaningless values) below for converting to MSV notation. Evaluate later
            line = line[:] + '|' + 'F' + '|' + '123'
            line = line.replace('YY', '0S')
            limit = line.rfind('|')
            new_limit = limit - 2
            new_line = "z" + line[:new_limit] + 'z' + line[new_limit:]
            source = re.findall(r'(?<=z).*?(?=x)', new_line)
            destination = re.findall(r'(?<=x).*?(?=z)', new_line)
            cars_present = len(source)
            remainder = total_cars - cars_present
            source_output = []
            destination_output = []
            for s in source:
                number_string = ''.join(re.findall(r'[\d]', s))
                direction_string = ''.join(re.findall(r'[a-zA-Z]', s[1:]))
                if direction_string == '': direction_string = '-1'
                binary_string_of_box = np.binary_repr(int(number_string), width=5)
                binary_string_of_box = ','.join(list(binary_string_of_box))
                direction_string = ','.join(list(np.binary_repr(direction_convert[direction_string], width=3)))
                source_output.append(binary_string_of_box)
                source_output.append(direction_string)
            source_output = ','.join(source_output)
            for d in destination:
                number_string = ''.join(re.findall(r'[\d]', d))
                direction_string = ''.join(re.findall(r'[a-zA-Z]', d[1:]))
                if direction_string == '': direction_string = '-1'
                binary_string_of_box = np.binary_repr(int(number_string), width=5)
                binary_string_of_box = ','.join(list(binary_string_of_box))
                direction_string = ','.join(list(np.binary_repr(direction_convert[direction_string], width=3)))
                destination_output.append(binary_string_of_box)
                destination_output.append(direction_string)
            destination_output = ','.join(destination_output)
            extra_for_source = (',1' * 8 * remainder)
            extra_for_destination = deepcopy(extra_for_source)
            action_number = str(actions_convert[line[limit - 1]])
            qvalue = line[limit + 1:]
            final_encoding = source_output + extra_for_source + ',' + destination_output + extra_for_destination + '|' + action_number + '|' + qvalue
            split_data = final_encoding.split('|')
            state_list = ''.join(split_data[0]).split(',')
            state_data = list(map(int, state_list))
            total_data.append(state_data)
        return np.array(total_data)

    def remember(self, sample):
        """
        Ths function handles the appending of the sample to the maintained deque. A sample consists of the state,
        action, next state and reward.
        :param sample: A tuple consisting of the above features
        :return:
        """
        self.memory.append(sample)
        self.episode_memory.append(sample)

    def replay(self, replay_given_state=None):
        """
        This function is responsible for processing the deque and then the extracted values are passed onto the neural
        network. Two neural networks are used. One is the target network that is used for calculating the true q values
        and the other is our main network used to calculate the predictions of the q values.

        1)  Create a minibatch by randomly sampling a batch size of elements from the memory deque.
        2)  Extracting the states, next_states, actions, rewards and masks from the minibatch.
        3)  Updating the new values from the target values.
            3a) Calculate the best action from the original model
            3b) Choose the qvalue of that action from the target model
            3c) Fitting the data

        :param replay_given_state: This argument is a special deque which is used when the neural network has to be
               trained for peculiar cases such as when an episode ends or when the car takes an illegal action.
               This consists of a single sample only representing that particular process.
        :return:
        """

        if replay_given_state is not None:
            minibatch = np.array(replay_given_state)[[0]]
            # positive_samples = np.array([sample for sample in self.memory if sample[3] < 0])
            # if positive_samples.shape[0]:
            #     minibatch = np.concatenate((minibatch, positive_samples), axis=0)
            print("minibatch", minibatch)
            self.epochs = 2

        else:
            # self.memory_logger.info('Memory before ' + str(self.memory))
            # self.memory = set(self.memory)
            # self.memory_logger.info('Memory after set ' + str(set(self.memory)))
            # self.memory_logger.info('Memory after deque ' + str(deque(self.memory)))
            # exit()
            """1"""

            self.memory = deque(OrderedSet(self.memory))
            # self.memory_logger.info('Memory after is ' + str(self.memory))
            if len(self.memory) < self.annealed_batch_size:
                return
            # self.memory = sorted(self.memory, key=lambda sample: hash(sample), reverse=False)
            self.epochs = 1
            idx = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
            # self.memory_logger.info('Memory is ' + str(self.memory))
            # exit()
            '''
            if self.epsilon >= 0.8:
                positive_samples = np.array([sample for sample in self.memory if sample[3] > 0])
                negative_samples = np.array([sample for sample in self.memory if sample[3] < 0])
                positive_size = len(positive_samples)
                negative_size = len(negative_samples)
                
                self.memory_logger.info('Number of positive samples are ' + str(positive_size))
                self.memory_logger.info('Number of negative samples are ' + str(negative_size))
                
                # size = min(positive_size, negative_size)
                idx_positive = np.random.choice(positive_size, size=positive_size, replace=False)
                idx_negative = np.random.choice(negative_size, size=negative_size - self.batch_size, replace=False)
                minibatch_negative = negative_samples[idx_negative]
                minibatch_positive = positive_samples[idx_positive]
                minibatch = np.concatenate((minibatch_positive, minibatch_negative), axis=0)
            else:
                idx = np.random.choice(len(self.memory), size=len(self.memory), replace=False)
                minibatch = np.array(self.memory)[idx]
            #    minibatch_positive = positive_samples[idx_positive]
            #if positive_size < self.batch_size // 2:
            #    size = self.batch_size if negative_size >= self.batch_size else negative_size
            #    idx_negative = np.random.choice(negative_size, size=size, replace=False)  
            #    minibatch = negative_samples[idx_negative]  
            #else:
            #    idx_positive = np.random.choice(positive_size, size=positive_size, replace=False)
                
            #    minibatch_negative = negative_samples[idx_negative]
            #    minibatch_positive = positive_samples[idx_positive]
                
            # self.memory = deque(set(self.memory))
            # buffer = sorted(self.memory, key=lambda replay: float(abs(replay[3])), reverse=True)
            # memory = len(buffer)
            # p = np.array([self.unusual_sample_factor ** i for i in range(len(buffer))])
            # p = p / sum(p)
            # if self.epsilon <= 0.3: size = len(self.memory)
            # idx = np.random.choice(np.arange(memory), size=size, p=p)
            
            # positive_minibatch = np.array(buffer)[idx]
            # p = np.array([(1-self.unusual_sample_factor) ** i for i in range(len(buffer))])
            # p = p / sum(p)
            # idx = np.random.choice(np.arange(memory), size=size // 2, p=p)
            # negative_minibatch = np.array(buffer)[idx]
            #positive_samples = np.array([sample for sample in self.memory if sample[3] > 0])
            #if positive_samples.shape[0]:
            #    minibatch = np.concatenate((minibatch, positive_samples), axis=0)
            # minibatch = np.concatenate((minibatch_positive, minibatch_negative), axis=0)
            '''
            minibatch = np.array(self.memory)[idx]
            self.memory_logger.info('Minibatch is ' + str(minibatch))

        """2"""

        np.random.shuffle(minibatch)
        masks = [self.masks[next_state] for next_state in minibatch[:, 2]]
        states = self.convert(list(minibatch[:, 0]), self.total_cars)
        actions = np.asarray(minibatch[:, 1], dtype=int)
        next_states = self.convert(list(minibatch[:, 2]), self.total_cars)
        rewards = np.asarray(minibatch[:, 3], dtype=float)

        # true = rewards + self.discount * np.amax(self.sess.run(self.target_model.y, feed_dict={self.x: next_states}),
        #                                          axis=1)
        
        ### Target dqn starts
        # target_predictions = self.target_model.predict(next_states)
        # max_qvalues = np.array([max(qvalues[masks[i]]) for i, qvalues in enumerate(target_predictions)])
        ### Target dqn ends
        
        ### Double dqn starts

        """3"""

        model_predictions = self.predict(next_states)
        max_actions = []
        invalid_count = 0
        """3a"""

        for i, qvalues in enumerate(model_predictions):
            #overall_max_qvalue = max(qvalues)
            #overall_max_action = int(np.argwhere(qvalues == overall_max_qvalue))
            overall_max_action = np.argmax(qvalues)
            #max_qvalue = max(qvalues[masks[i]])
            max_action = np.argmax(qvalues[masks[i]])
            max_actions.append(max_action)
            if overall_max_action != max_action:
                invalid_count +=1
                # self.invalid_action_logger.info('For epsilon ' + str(self.epsilon) + ' Invalid action ' + str(overall_max_action) + ' Valid action ' + str(max_action))
        max_actions = np.array(max_actions)
        size = len(minibatch)
        if invalid_count > 0:
            self.invalid_action_logger.info('For epsilon ' + str(self.epsilon) + ' Invalid actions count ' + str(invalid_count) + ' Valid action count ' + str(size - invalid_count) + ' Total invalid to memory ratio is ' + str(invalid_count / size )) 
        # max_actions = np.array([int(np.argwhere(qvalues == max(qvalues[masks[i]]))[0]) for i, qvalues in enumerate(model_predictions)])
        """3b"""

        max_qvalues = np.choose(max_actions, self.target_predict(next_states).T)
        ### Double dqn ends
        
        true = rewards + self.discount * max_qvalues
        # true = rewards + self.discount * np.choose(np.argmax(self.model.predict(next_states), axis=1), self.target_model.predict(next_states).T)
        # predictions = self.sess.run(self.model.y, feed_dict={self.x: states})
        predictions = self.predict(states)
        # Update the network's predictions with the new predictions we have.
        for i in range(len(predictions)):
            # Flag states as terminal (the last state before a epoch ended).
            # terminal_state = (next_states[i] == np.array([None] * self.input_dim)).all()
            # Update each state's Q-value prediction with our new estimate.
            # Terminal states have no future, so set their Q-value to their immediate reward.
            predictions[i][actions[i]] = true[i]

        # Propagate the new predictions through our network.
        x_train, y_train = states, predictions
        # x_val, y_val = deepcopy(x_train), deepcopy(y_train)
        """3c"""

        self.sess.run(self.optimizer, feed_dict={
            self.output_tensor: y_train,
            self.input_tensor: x_train,
        })

    def initial_configuration_memory(self, no_of_cars, destination_index_array, input_car_states, destination_choose_arrays):
        """
        This function adds the initial state samples inolving the best action for every car for all 4 orientations.
        This was done to ensure that our model would not forget the starting state samples after training begins.
        :param no_of_cars: Total number of cars
        :param destination_index_array: Index array required for choosing the starting destination
        :param destination_choose_arrays: Choose array required for choosing the specific destinations to be visited
        :param input_car_states:  The car state
        :return:
        """
        self.initial_memory = deque(maxlen=self.mem_size)
        env.reset()
        env.initialize_cars(no_of_cars)
        env.select_reward_function('dqn_reward')
        env.initialize(destination_index_array, input_car_states, destination_choose_arrays)
        for num, i in enumerate(env.get_cars()):
            state = env.encodestate(num, 1)
            for direction in ['N', 'E', 'W', 'S']:
                state = state[:2] + direction + state[3:]
                temp_check_state = state.replace(direction, '')
                if temp_check_state in env.bad_states:
                    return
                _, valid_actions = self.give_mask(state)
                data = []
                for action in valid_actions:
                    next_state = env.get_successor_state(deepcopy(state), action)
                    temp_check_ns = next_state.replace(direction, '')
                    if temp_check_ns in env.bad_states: continue
                    reward = env.reward_function(deepcopy(state), deepcopy(next_state), num, action)
                    data.append((next_state, reward, action))
                data = sorted(data, reverse=True, key=lambda tupl: tupl[1])
                if data[0][1] < 0 or data[0][1] == 1:
                    continue
                final = [data[i] for i, (ns, reward, action) in enumerate(data) if reward == data[0][1]]
                if len(final) > 1: continue
                for data in final:
                    sample = (state, self.actions_dict[data[2]], data[0], data[1])
                    if sample[1] == 3: break
                    self.initial_memory.append(sample)
                    self.masks[state] = self.give_mask(state)[0]
                    self.masks[sample[2]] = self.give_mask(sample[2])[0]
                
    def handle_special_states(self, sample):
        self.remember(sample)
        self.state_logger.info('Negative Sample is ' + str(sample))
        negative_sample_memory = deque()
        negative_sample_memory.append(sample)
        self.replay(negative_sample_memory)
        self.state_logger.info(str(sample))

    def train_given_state(self, episodes, inputfile_number, outputfile_number, input_car_states=None, no_of_cars=1,
                          destination_index_array=[0, 0, 0, 0], destination_choose_arrays=[0, 0, 0], starting_episode=1):
        """
        This function consists of the complete training procedure involved for training a particular state.

        1) Initialize memory with initial_configuration_memory and create episode_memory and special_memory
        2) Loop is run for total number of episodes and cars are allowed to taken an action in ordered manner
            2a) For every episode, get a sample (state, action, next state and reward)
            2b) Store sample in a deque and append sample to memory and episode memory
            2c) Run the fit function algorithm after every iteration in a single episode.

        3) Check if episode memory + special_memory is less than batch_size
            3a) If True, add episode_memory to special_memory else replace special_memory with episode_memory
        4) Check and update target model at every 50 episodes.

        :param episodes: Total number of episodes for which the state will be trained
        :param destination_index_array: Index array required for choosing the starting destination
        :param destination_choose_arrays: Choose array required for choosing the specific destinations to be visited
        :param input_car_states:  The car state
        :param starting_episode:  The episode number from where we begin the training
        :param no_of_cars:  The car state
        :return:
        """

        """1"""
        dir = os.path.dirname(os.path.realpath(__file__))

        model_name = 'model-0.' + str(datetime.datetime.now()) + '.ckpt'
        self.save_model(dir[:len(dir) - 10] + MODEL_DIRECTORY + '/' + model_name)
        if self.load_model_flag[0]:
            print("Loading previous model")
            self.model = load_model(dir[:len(dir) - 10] + MODEL_DIRECTORY + '/' + self.load_model_flag[1], custom_objects={'huber_loss': huber_loss})

        self.update_target_model()
        self.env_config.no_of_cars = no_of_cars
        self.epsilon = 1
        self.initial_configuration_memory(no_of_cars, destination_index_array, input_car_states,
                                          destination_choose_arrays)
        self.memory_logger.info('Initial memory is ' + str(self.initial_memory))
        """2"""
        for episode in range(starting_episode, episodes + 1):
            self.state_logger.info('Episode ' + str(episode) + ' beginning')
            self.memory_logger.info('Episode ' + str(episode) + ' beginning')
            self.memory = deque(maxlen=self.mem_size)
            self.episode_memory = deque(maxlen=self.mem_size)
            self.memory = deepcopy(self.initial_memory)
            if self.special_memory:
                print("length of special memory", len(self.special_memory))
                self.memory += self.special_memory

            print('episode :: ', episode)
            total_reward = 0
            self.epsilon -= float(1 / episodes)
            env.reset()
            env.initialize_cars(no_of_cars)
            env.select_reward_function('dqn_reward')
            """
            Defining initial sources and destinations for cars.
            We are also storing defined source into 'given_nodes' list.
            Which will be used later in exploitation part.
            """

            env.initialize(destination_index_array, input_car_states, destination_choose_arrays)
            print('episode', episode, self.epsilon, 'annealed_batch_size', self.annealed_batch_size)
            self.training_done = 0
            counts_in_one_episode = 0
            self.episode_array = []
            while not self.training_done:
                counts_in_one_episode += 1
                # self.random_shuffle_car_map()
                for num, i in enumerate(env.get_cars()):
                    if env.check_car_training(num):
                        continue
                    state = env.encodestate(num, 1)
                    action, action_number = self.getaction(state)
                    self.episode_array.append(action)
                    if len(self.episode_array) > 10000:  # Added this for non-solvable states
                        print('\nSkipping! The state :: %s cannot be trained as it has no paths possible.' % state)
                        return 0
                    next_state = env.get_successor_state(deepcopy(state), action)
                    if self.epsilon <= 0:
                        print(state, action, next_state)
                    reward = env.reward_function(deepcopy(state), deepcopy(next_state), num, action)
                    next_state_array = env.state_to_array(next_state)
                    env.update_car(num, deepcopy(next_state_array[0][0]), deepcopy(next_state_array[0][1]), 0)
                    self.masks[state] = self.give_mask(state)[0]
                    self.masks[next_state] = self.give_mask(next_state)[0]
                    """2a"""
                    sample = (state, action_number, next_state, reward)

                    """2b"""
                    self.remember(sample)
                    """2c"""
                    self.replay()

                    if self.epsilon <= 0:
                        # env.update_simulator_display(2)
                        pass

                    total_reward += reward
                    self.state_logger.info(str(sample))
                self.training_done = env.is_game_ended()

            print("Episode: {}, Total reward: {}, Explore P: {}".format(episode, total_reward, self.epsilon))
            print("length of memory :", len(self.memory))

            """
             The below line is used for updating the target neural network weights with the main neural network weights
             This is known as fixed target dqn. 
             """

            """3a"""
            self.episode_memory = deque(set(self.episode_memory))
            if len(self.episode_memory) < self.batch_size:
                if len(self.special_memory) + len(self.episode_memory) < self.batch_size:
                    self.special_memory += self.episode_memory
                else:
                    self.special_memory = self.episode_memory

            """4"""
            if episode % 50 == 0:
                self.update_target_model()
                self.state_logger.info('50 episodes complete---------------------------------------')
                print("Saving Model")
                model_name = 'model-' + str(episode) + '.' + str(datetime.datetime.now()) + '.ckpt'
                self.save_model(dir[:len(dir) - 10] + MODEL_DIRECTORY + '/' + model_name)
                if episode == episodes:
                    self.load_model_flag = (1, model_name)
                #     self.model.compute_fisher(x_train, self.sess, len(x_train))
                #     self.model.star()
            self.state_logger.info('Episode ' + str(episode) + ' ending')
            self.memory_logger.info('Episode ' + str(episode) + ' ending')

    def train_given_state_random_order(self, episodes, inputfile_number, outputfile_number, input_car_states=None, no_of_cars=1,
                          destination_index_array=[0, 0, 0, 0], destination_choose_arrays=[0, 0, 0], starting_episode=1):

        """
        This function consists of the complete training procedure involved for training a particular state.

        1) Initialize memory with initial_configuration_memory and create episode_memory and special_memory
        2) Loop is run for total number of episodes and any randomly chosen car is allowed to take an action
            2a) For every episode, get a sample (state, action, next state and reward)
            2b) Store sample in a deque and append sample to memory and episode memory
            2c) Run the fit function algorithm after every iteration in a single episode.

        3) Check if episode memory + special_memory is less than batch_size
            3a) If True, add episode_memory to special_memory else replace special_memory with episode_memory
        4) Check and update target model at every 50 episodes.

        :param episodes: Total number of episodes for which the state will be trained
        :param destination_index_array: Index array required for choosing the starting destination
        :param destination_choose_arrays: Choose array required for choosing the specific destinations to be visited
        :param input_car_states:  The car state
        :param starting_episode:  The episode number from where we begin the training
        :param no_of_cars:  The car state
        :return:
        """

        """1"""
        dir = os.path.dirname(os.path.realpath(__file__))
        # self.tb_call_back = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True)
        model_name = 'model-0.' + str(datetime.datetime.now()) + '.h5'
        self.model.save(dir[:len(dir) - 10] + MODEL_DIRECTORY + '/' + model_name)
        # model_name = 'model-0.' + str(datetime.datetime.now()) + '.h5'
        # self.model.save(dir[:len(dir) - 10] + 'models/' + model_name)
        # self.model.restore(self.sess)
        # saver = tf.train.Saver(self.model.var_list)
        # model_name = str('model-0' + datetime.datetime.now())
        # model_name = 'model-0'
        # saver.save(self.sess, dir[:len(dir)-10] + 'models/' + model_name)
        if self.load_model_flag[0]:
            print("Loading previous model")
            self.model = load_model(dir[:len(dir) - 10] + MODEL_DIRECTORY + '/' + self.load_model_flag[1], custom_objects={'huber_loss': huber_loss})
            # new_saver = tf.train.import_meta_graph(dir + '/models/' + self.load_model_flag[1] + '.meta')
        #     new_saver.restore(self.sess, tf.train.latest_checkpoint(dir + '/models/'))
        #     self.update_target_model()
        # self.model = load_model('models/fixed_target_single_car.h5')
        self.update_target_model()
        self.env_config.no_of_cars = no_of_cars
        self.epsilon = 1
        self.initial_configuration_memory(no_of_cars, destination_index_array, input_car_states, destination_choose_arrays)
        self.memory_logger.info('Initial memory is ' + str(self.initial_memory))
        for episode in range(starting_episode, episodes + 1):
            #if episode == 0.85 * episodes:
            #    self.state_logger.info('Train given state random order beginning')
            #    self.train_given_state_random_order(episodes, inputfile_number, outputfile_number, input_car_states, no_of_cars, destination_index_array, destination_choose_arrays, episode)
            #    return
            self.state_logger.info('Episode ' + str(episode) + ' beginning')
            self.memory_logger.info('Episode ' + str(episode) + ' beginning')
            self.memory = deque(maxlen=self.mem_size)
            self.episode_memory = deque(maxlen=self.mem_size)
            self.memory = deepcopy(self.initial_memory)
            if self.special_memory:
                print("length of special memory", len(self.special_memory))
                self.memory += self.special_memory
                # self.special_memory = None
                # print("in cond", self.memory)
            print('episode :: ', episode)
            # # self.terminal_logger.info('Episode Number ' + str(episode))
            total_reward = 0
            # if episode <= 0.4 * episodes:
            # self.epsilon = give_action_choose_probability_new(episodes, episode, self.probability_model)
            #else:
            #    self.epsilon -= float(0.8 / episodes)
            #self.annealed_batch_size = give_batch_size_choose_probability(episodes, episode, self.probability_model, self.batch_size)
            self.epsilon -= float(1 / episodes)
            env.reset()
            env.initialize_cars(no_of_cars)
            env.select_reward_function('dqn_reward')
            """
            Defining initial sources and destinations for cars.
            We are also storing defined source into 'given_nodes' list.
            Which will be used later in exploitation part.
            """

            env.initialize(destination_index_array, input_car_states, destination_choose_arrays)
            print('episode', episode, self.epsilon, 'annealed_batch_size', self.annealed_batch_size)
            # self.terminal_logger.info('Episode ' + str(episode) + ' Epsilon ' + str(self.epsilon))
            self.training_done = 0
            counts_in_one_episode = 0
            self.episode_array = []
            while not self.training_done:
                counts_in_one_episode += 1
                # self.random_shuffle_car_map()
                num = random.randint(0, env.no_of_cars - 1)
                # num0 = self.car_one_to_one_map_dict[num]
                # print(num, i, env.check_car_training(num))
                # input()
                if env.check_car_training(num):
                    continue
                # self.encode_state_logger.info('Entering Encode State')
                state = env.encodestate(num, 1)
                # self.encode_state_logger.info('Exiting Encode State')
                action, action_number = self.getaction(state)
                # self.terminal_logger.info('Action is ' + action)
                # self.encode_state_logger.info('Flag for action not in legal states is ' + str(self.flag))
                self.episode_array.append(action)
                if len(self.episode_array) > 10000:  # Added this for non-solvable states
                    print('\nSkipping! The state :: %s cannot be trained as it has no paths possible.' % state)
                    return 0
                next_state = env.get_successor_state(deepcopy(state), action)
                if self.epsilon <= 0:
                    print(state, action, next_state)
                reward = env.reward_function(deepcopy(state), deepcopy(next_state), num, action)
                self.masks[state] = self.give_mask(state)[0]
                self.masks[next_state] = self.give_mask(next_state)[0]
                """2a"""
                sample = (state, action_number, next_state, reward)
                # self.encode_state_logger.info('Sample is ' + str(sample))
                """2b"""
                self.remember(sample)
                """2c"""
                self.replay()
                # This tells us about early stopping where we can stop training the model
                # if self.callbacks_list[0].wait >= self.callbacks_list[0].patience:
                #     break
                # print("dqn working okay")
                # self.update(state, action, next_state,
                #             reward)
                next_state_array = env.state_to_array(next_state)
                env.update_car(num, deepcopy(next_state_array[0][0]), deepcopy(next_state_array[0][1]), 0)
                if self.epsilon <= 0:
                    # self.qval_logger.info(
                    #     'For episode ' + str(episode) + ' Qvalues for sample ' + str(sample) + ' is ' + str(
                    #         self.sess.run(self.model.y,
                    #                       feed_dict={self.x: self.convert([state], self.total_cars)})[0]))
                    # self.qval_logger.info('----------------------------------------')
                    pass
                    # env.update_simulator_display(2)
                # if episode % 10 == 0:
                #     print("Current memory: ", self.memory)
                total_reward += reward
                self.state_logger.info(str(sample))
                # print(state, next_state, reward, total_reward)
                self.training_done = env.is_game_ended()
            # print("going to replay", deque().append(sample))
            # self.replay(deque().append(sample))
            # print('counts_in_one_episode', counts_in_one_episode)
            print("Episode: {}, Total reward: {}, Explore P: {}".format(episode, total_reward, self.epsilon))
            # self.terminal_logger.info('Episode ' + str(episode) + ' Total Reward ' + str(total_reward) + ' Epsilon ' + str(self.epsilon))
            print("length of memory :", len(self.memory))
            # self.terminal_logger.info('Length of memory ' + str(len(self.memory)))
            # self.terminal_logger.info('Updating Target Model')

            """
             The below line is used for updating the target neural network weights with the main neural network weights
             This is known as fixed target dqn. 
             """

            """3a"""
            self.episode_memory = deque(set(self.episode_memory))
            if len(self.episode_memory) < self.batch_size:
                if len(self.special_memory) + len(self.episode_memory) < self.batch_size:
                    self.special_memory += self.episode_memory
                else: self.special_memory = self.episode_memory

            # elif episode % 2 == 0:
            # self.qval_logger.info('For episode ' + str(episode) + ' Qvalues for sample ' + str(sample) + ' is ' + str(
            #     self.model.predict(self.convert(['U6NxU2WzU8xU6'], self.total_cars))[0]))
            # self.qval_logger.info('For episode ' + str(episode) + ' Qvalues for sample ' + str(sample) + ' is ' + str(
            #     self.model.predict(self.convert(['U8WxU6WzU6xU2'], self.total_cars))[0]))

            """4"""
            if episode % 50 == 0:
                self.update_target_model()
                self.state_logger.info('50 episodes complete---------------------------------------')
                print("Saving Model")
                model_name = 'model-' + str(episode) + '.' + str(datetime.datetime.now()) + '.h5'
                # weights = self.model.var_list
                # saver = tf.train.Saver(weights)
                # model_name = str(datetime.datetime.now())
                # model_name = 'model'
                # saver.save(self.sess, dir[:len(dir) - 10] + 'models/' + model_name, global_step=50)
                self.model.save(dir[:len(dir)-10] + MODEL_DIRECTORY + '/' + model_name)
                if episode == episodes:
                    self.load_model_flag = (1, model_name)
                #     self.model.compute_fisher(x_train, self.sess, len(x_train))
                #     self.model.star()
            self.state_logger.info('Episode ' + str(episode) + ' ending')
            self.memory_logger.info('Episode ' + str(episode) + ' ending')

    def detect_deadlocks_v0(self, episodes=40, inputfile_number=0, outputfile_number=0, input_car_states=None, no_of_cars=1, destination_index_array=[0, 0, 0, 0], destination_choose_arrays=[0, 0, 0]):
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
        env.no_of_cars = no_of_cars
        for episode in range(0, episodes):
            print(episode, "episode")
            self.epsilon = give_action_choose_probability(episodes, episode, self.probability_model)
            env.reset()
            env.select_reward_function('dqn_reward')
            """
            Defining initial sources and destinations for cars.
            We are also storing defined source into 'given_nodes' list.
            Which will be used later in exploitation part.
            """
            # if input_car_states is not None:
            #     given_nodes = []
            #     for num,i in enumerate(self.cars):
            #         nodes = deepcopy(env.box_node_list[input_car_states[num]])
            #         random.shuffle(nodes)
            #         self.cars[num].source_node = nodes[0]
            #         self.cars[num].destination_index = destination_index_array[num]
            #         self.cars[num].destination_choose_array = destination_choose_arrays[num]
            #         given_nodes.append(nodes[0])
            env.initialize(destination_index_array, input_car_states, destination_choose_arrays)
            print('episode', episode, self.epsilon)
            self.training_done = 0
            counts_in_one_episode = 0
            self.episode_array = []
            # self.blocked_nodes_list = []
            deadlock_count = 0  # TODO: EXTRA
            deadlock_detected = 0  # TODO: EXTRA
            while not self.training_done:
                if deadlock_detected == 1:
                    print("deadlock detected")

                    break
                counts_in_one_episode += 1
                for num, i in enumerate(env.get_cars()):
                    # num0 = self.car_one_to_one_map_dict[num]
                    if env.check_car_training(num):
                        continue
                    state = env.encodestate(num, 0)
                    action = self.getaction(state, episode)

                    # print ("encoded_state", state, "action_given", action)
                    if action == 'S' and self.epsilon == 0:
                        deadlock_count += 1
                        if env.no_of_cars <= 2:
                            if deadlock_count == env.no_of_cars * 2:
                                print("deadlock_detected")
                                env.update_simulator_display(3000)
                                save_deadlock_state(state, env.destination_box_list, env.environment_bad_states_file, env.no_of_cars)
                                deadlock_detected = 1
                                break
                        else:
                            if deadlock_count == env.no_of_cars:
                                print("deadlock_detected")
                                env.update_simulator_display(3000)
                                save_deadlock_state(state, env.destination_box_list, env.environment_bad_states_file,
                                                    env.no_of_cars)
                                deadlock_detected = 1
                                break
                    else:
                        deadlock_count = 0
                    self.episode_array.append(action)
                    # print(action, "Action from detect deadlock method")
                    next_state = env.get_successor_state(deepcopy(state), action)

                    if len(self.episode_array) > 10000:  # Added this for non-solvable states
                        print('\nSkipping! The state :: %s cannot be trained as it has no paths possible.' % state)
                        return False
                    if self.epsilon <= 0:
                        print(state, action, next_state)
                    next_state_array = env.state_to_array(next_state)
                    env.update_car(num, deepcopy(next_state_array[0][0]), deepcopy(next_state_array[0][1]), 0)
                    if self.epsilon <= 0:
                        env.update_simulator_display(100)
                self.training_done = env.is_game_ended()
            print('counts_in_one_episode', counts_in_one_episode)
            if deadlock_detected == 1:
                return episode
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
        # f = open("data/qvalue_files/up_right.txt", 'w', encoding='utf-8')
        self.env_config.no_of_cars = no_of_cars
        env.no_of_cars = no_of_cars
        for i in range(0, episodes):
            total_cars = 5
            # self.__init__(8 * total_cars * 2, 5, total_cars, self.sess)
            # self.__init__(8 * total_cars * 2, 5, total_cars, env_config=self.env_config, sess=self.sess)

            self.load_model(self.sess, './' + inputfile_number + '/')

            """
            Defining initial sources and destinations for cars.
            """
            env.reset()
            env.initialize_cars(no_of_cars)
            env.initialize(destination_index_array, input_car_states, destination_choose_arrays)
            counts_in_one_episode = 0
            self.episode_array = []
            # self.blocked_nodes_list = []
            self.validation_done = 0
            self.epsilon = 0
            while not self.validation_done:
                counts_in_one_episode += 1
                num = env.update_simulator_display(waitkey) - 48
                print("number passes by key press", num)
                if num >= env.no_of_cars:
                    print("in valid car id")
                    continue
                if env.check_car_training(num):
                    continue
                # print (i.source_node, "source_node")
                state = env.encodestate(num, 1)
                action, action_index = self.getaction(state, 0)
                # f.write(state + '|' + action + '|\n')
                print("for state,", state, "action provided", action)
                next_state = env.get_successor_state(deepcopy(state), action)
                next_state_array = env.state_to_array(next_state)
                env.update_car(num, deepcopy(next_state_array[0][0]), deepcopy(next_state_array[0][1]))
                env.update_simulator_display(waitkey)
                self.validation_done = env.is_game_ended()
                if self.validation_done:
                    break


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
