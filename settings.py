import os

# The directory in which the tensorflow models will be stored if using DQN
MODEL_DIRECTORY = 'models'
if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)
TOTAL_CARS = 5

# Number of episodes to run the algorithm for
EPISODES = 5000

# Input dimension of the neural-network for any particular state
INPUT_DIM = 8 * TOTAL_CARS * 2

# Output dimension of the neural network representing Q-values for DQN
OUTPUT_DIM = 5

LEARNING_RATE = 0.0001

# Batch Size used for training using DQN
BATCH_SIZE = 32

# Seed to be used for reproducibility
SEED = 3

# Name of the algorithm file that is defined in the src/algorithms directory
algorithm = 'dqn_open_source'

# Parameters for the RL class defined in the above algorithm .py file
params = {
    'input_dim': INPUT_DIM,
    'output_dim': OUTPUT_DIM,
    'cars': TOTAL_CARS,
    'batch_size': BATCH_SIZE
}
