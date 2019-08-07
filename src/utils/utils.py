import random
import settings
import itertools as it
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow import keras
import numpy as np
random.seed(settings.SEED)
tf.set_random_seed(settings.SEED)
from collections import defaultdict

def give_action_choose_probability(episodes, episode, model):
    """
    This method gives probability of choosing action according to the episode number and total number of episode.
    :param episodes: total number of episodes
    :param episode: current running episode
    :param model: model number
    :return: probability
    """
    if model == 0:
        # pure qvalue defined model, no randomness
        return 0
    if model == 1:
        # pure random model.
        return 1
    if model == 2:
        # if episode<40% random else not random
        threshold = int(episodes * 0.4)
        if episode < threshold:
            return 1
        else:
            return 0
    if model == 3:
        # if episode<40% random, if between 40 to 50 0.8, if 50 to 70, 0.5, if between 70 to 85 0.3, if between 85 to 100 0
        threshold0 = int(episodes * 0.4)
        threshold1 = int(episodes * 0.6)
        threshold2 = int(episodes * 0.7)
        threshold3 = int(episodes * 0.85)
        threshold4 = int(episodes * 0.99)
        threshold5 = int(episodes * 1)
        if episode < threshold0:
            return 1
        elif episode < threshold1:
            return 0.8
        elif episode < threshold2:
            return 0.5
        elif episode < threshold3:
            return 0.3
        elif episode < threshold4:
            return 0.15
        else:
            return 0


def give_action_choose_probability_new(episodes, episode, model):
    """
    This method gives probability of choosing action according to the episode number and total number of episode.
    :param episodes: total number of episodes
    :param episode: current running episode
    :param model: model number
    :return: probability
    """
    if model == 0:
        # pure qvalue defined model, no randomness
        return 0
    if model == 1:
        # pure random model.
        return 1
    if model == 2:
        # if episode<40% random else not random
        threshold = int(episodes * 0.4)
        if episode < threshold:
            return 1
        else:
            return 0
    if model == 3:
        # if episode<40% random, if between 40 to 50 0.8, if 50 to 70, 0.5, if between 70 to 85 0.3, if between 85 to 100 0
        threshold0 = int(episodes * 0.6)
        threshold1 = int(episodes * 0.8)
        threshold2 = int(episodes * 0.9)
        # threshold3 = int(episodes * 0.85)
        threshold4 = int(episodes * 0.99)
        threshold5 = int(episodes * 1)
        if episode < threshold0:
            return 1
        elif episode < threshold1:
            return 0.8
        elif episode < threshold2:
            return 0.5
        # elif episode < threshold3:
        #    return 0.3
        elif episode < threshold4:
            return 0.15
        else:
            return 0


def give_batch_size_choose_probability(episodes, episode, model, original=32):
    """
    This method gives probability of choosing action according to the episode number and total number of episode.
    :param episodes: total number of episodes
    :param episode: current running episode
    :param model: model number
    :return: probability
    """
    if model == 0:
        # pure qvalue defined model, no randomness
        return 0
    if model == 1:
        # pure random model.
        return 1
    if model == 2:
        # if episode<40% random else not random
        threshold = int(episodes * 0.4)
        if episode < threshold:
            return 1
        else:
            return 0
    if model == 3:
        # if episode<40% random, if between 40 to 50 0.8, if 50 to 70, 0.5, if between 70 to 85 0.3, if between 85 to 100 0
        threshold0 = int(episodes * 0.4)
        threshold1 = int(episodes * 0.5)
        threshold2 = int(episodes * 0.7)
        threshold3 = int(episodes * 0.85)
        threshold4 = int(episodes * 0.99)
        threshold5 = int(episodes * 1)
        if episode < threshold0:
            return original
        elif episode < threshold1:
            return original
        elif episode < threshold2:
            return original
        elif episode < threshold3:
            return original//2
        elif episode < threshold4:
            return original//4
        else:
            return original//8


def calculated_deadlock_states(state, generated_deadlock_boxes):
    """
    This function takes in a list of deadlock boxes and a deadlock state and adds a station_box at the end of each of
    the state.
    :param generated_deadlock_boxes: list of boxes that are to be appended at the end of each string
    :param state: deadlock state string
    :return: list of deadlocks
    """
    array = state.split('z')
    deadlocks_list = []
    source_boxes = [x[:2] for x in array]
    for deadlock_box in generated_deadlock_boxes:
        if deadlock_box in source_boxes:
            continue
        deadlocks_list.append(state + 'z' + deadlock_box + 'x' + deadlock_box)
    return deadlocks_list


def save_deadlock_state(state, generated_deadlock_boxes, bad_states_file, no_of_cars, max_cars):
    """
    This function appends the deadlock states
    :param generated_deadlock_boxes: list of boxes that are to be appended at the end of each string
    :param state: deadlock state string
    :return: list of deadlocks
    """
    for i in ['N', 'E', 'W', 'S']:
        state = state.replace(i, '')
    generated_deadlocks_list = [state]
    flag = defaultdict(int)

    for _ in range(1, max_cars - no_of_cars + 1):
        temp = []
        for state in generated_deadlocks_list:
            if not flag[state]:
                temp += calculated_deadlock_states(state, generated_deadlock_boxes)
            flag[state] = 1
        generated_deadlocks_list += temp

    generated_deadlocks_list = list(set(generated_deadlocks_list))
    with open(bad_states_file, 'r') as b:
        stored_deadlocks = b.read().split()

    to_be_added_deadlocks = []
    for state in generated_deadlocks_list:
        array = state.split('z')
        combinations = list(it.permutations(array, len(array)))
        for i in combinations:
            new = i[0].split('x')
            if new[0] == new[1]:
                continue
            state = ''
            for j in i:
                state += 'z' + j
            state = state[1:]
            to_be_added_deadlocks.append(state)
    final_deadlocks = list(set(stored_deadlocks + to_be_added_deadlocks))
    with open(bad_states_file, 'w+') as f:
        for state in final_deadlocks:
            f.write(state + '\n')


def noisy_dense(inputs, units, bias_shape, c_names, w_i, b_i=None, activation=tf.nn.relu,
                noisy_distribution='factorised'):
    """
    This function creates tensor variables for weights, weights_noises, bias and bias_noises and reuses them if already
    created. These variables are used during training. Variables are created based on type of noisy_distribution.

    :param inputs: Input tensor
    :param units: Number of units in a hidden layer
    :param bias_shape: Shape of the bias layer according to the number of units
    :param c_names: collection names
    :param w_i: weight initializer
    :param b_i: bias initializer
    :param activation: activation function to be used
    :param noisy_distribution: type of noisy dqn
    :return: list of deadlocks
    """

    def f(e_list):
        return tf.multiply(tf.sign(e_list), tf.pow(tf.abs(e_list), 0.5))
    if not isinstance(inputs, ops.Tensor):
        inputs = ops.convert_to_tensor(inputs, dtype='float')
        # dim_list = inputs.get_shape().as_list()
        # flatten_shape = dim_list[1] if len(dim_list) <= 2 else reduce(lambda x, y: x * y, dim_list[1:])
        # reshaped = tf.reshape(inputs, [dim_list[0], flatten_shape])
    if len(inputs.shape) > 2:
        inputs = tf.contrib.layers.flatten(inputs)
    flatten_shape = inputs.shape[1]
    weights = tf.get_variable('weights', shape=[flatten_shape, units], initializer=w_i)
    w_noise = tf.get_variable('w_noise', [flatten_shape, units], initializer=w_i)
    if noisy_distribution == 'independent':
        weights += tf.multiply(tf.random_normal(shape=w_noise.shape), w_noise)
    elif noisy_distribution == 'factorised':
        noise_1 = f(tf.random_normal(tf.TensorShape([flatten_shape, 1]), dtype=tf.float32))
        noise_2 = f(tf.random_normal(tf.TensorShape([1, units]), dtype=tf.float32))
        weights += tf.multiply(noise_1 * noise_2, w_noise)
    dense = tf.matmul(inputs, weights)
    if bias_shape is not None:
        assert bias_shape[0] == units
        biases = tf.get_variable('biases', shape=bias_shape, initializer=b_i)
        b_noise = tf.get_variable('b_noise', [1, units], initializer=b_i)
        if noisy_distribution == 'independent':
            biases += tf.multiply(tf.random_normal(shape=b_noise.shape), b_noise)
        elif noisy_distribution == 'factorised':
            biases += tf.multiply(noise_2, b_noise)
        if activation is not None:
            return tf.contrib.layers.layer_norm(activation(dense + biases))
        else:
            return tf.contrib.layers.layer_norm(dense + biases)


# The following code was taken from the UC-Berkeley CS-188 Pacman Course
# http: // inst.eecs.berkeley.edu / ~cs188 / pacman / pacman.html

def flipcoin(p):
    """
    :param p: float in [0,1]
    :return: Boolean {True, False}
    1. True when p=1
    2. False when p=0
    3. random True or False when p belongs to (0,1)
    The number of True decreases with decrease in p.
    """
    r = random.random()
    return r < p


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