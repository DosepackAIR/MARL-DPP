# Deep-Navigation

This project was built to conduct research on Navigation among multiple cars (agents) on a 3x3 square grid using 
Reinforcement Learning. Two RL approaches were used to solve this problem namely Q-learning and Noisy Double DQN.

The environment consists of 2 floors (up and down) and each floor is a 3x5 2d maze. Each floor
has been further divided into two 3x3 blocks (the 3rd column is common for both blocks).


## Getting Started

### Prerequisites

In your Python-3 virtual-environment run the following command

```
pip install -r requirements.txt
```

## Running the experiment
Basic Command

```
python app.py -i 0 -o 0 -dd 0 -v 0 -l 3 -bn 0
```

### app.py

This is the main file required to run the experiments. It assumes a few arguments that are to be entered by the
    user.

    1) -i (input file):
            This argument contains the name of the file which could either contain a saved tensorflow checkpoint file
            (DQN) or a qvalue .txt file (Q-learning) that is to be loaded prior to the training process.
            Default argument is 0 and can be any string. Its also used for validating a training process
            explained below. The .txt file for Q-learning is defined in the data/qvalue directory.

    2) -o (output file):
            This argument contains the name of the file which serves as the name that is used for saving tensorflow
            models (DQN) or a .txt file in the case of Q-learning.
            Tensorflow models are saved in the MODEL_DIRECTORY specified in settings.py for DQN.
            For Q-learning, the .txt file will be stored in the respective layout folder in the data/qvalue/ directory
            Default argument is 0

    3) -dd (detect deadlock):
            This argument is an integer specifying whether Q-learning should calculate the Deadlock states prior to
            beginning the training. This feature is available only for Q-learning.
            Default argument is 0, 1 for activating.

    4) -bn (block number):
            This argument is an integer specifying the block that has to be used for training for a particular layout.
            Values range from 0, 1, 2, 3.

            0 - up_left
            1 - down_left
            2 - down_right
            3 - up_right

            Currently blocks 0 and 2 have been implemented in layout.py

     5) -l (layout number):
             This argument is an integer specifying the layout that has to be used for training. Defined in layout.py
             Value can be any layout defined in layout.py

     6) -v (validate):
             This argument is an integer which allows us to check the performance of our trained model/ qvalue file.
             It Gives us a visual through an open-cv window and allows us to move the agents using num keys.
             HotKey press starts from 0 to n-1 for n agents (cars)
             Default argument - 0, 1 used for activating validation. Uses input file for loading model/ qvalue file.
             Input file has to be a .ckpt (without .ckpt in the arguments) file for loading the tensorflow model

## algorithms.py:
    This directory holds all the algorithms that can be run for training. Custom algorithms have to be defined here or
    dqn_open_source.py/ qlearning.py can be used for training an agent using DQN and Q-learning respectively.


## settings.py:
    This file contains important parameters/ hyper-parameters that are used throughout the training process. It
    should also contain the name of the algorithm file along with its parameter dictionary corresponding to the RL class
    defined inside every algorithm file ( see dqn_open_source.py or qlearning.py for reference )


## Acknowledgments

* The inspiration for this project was derived from UC Berkeley CS-188 RL course and the Nature DQN paper 

