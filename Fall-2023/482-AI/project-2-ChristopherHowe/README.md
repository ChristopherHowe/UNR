[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/KBLq5xA7)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13187823&assignment_repo_type=AssignmentRepo)

# project[2] - Reinforcement Learning
due 12/13/2023 midnight
(won't be considered late up to 12/19/2023)
Work to be done individually, group discussion is allowed

## Objectives:
To implement a reinforcement learning algorithm that can learn a policy for a given task based on task-based rewards
To take a continuous environment and discretize it so that it is suitable for a reinforcement learning task

This is the CartPole task. The idea here is to balance this pole using a one-dimensional robot (it can only move left and right). The robot's state has 4 components:
x: the location of the robot (0 is the center, -2.4 is the leftmost part of the board, 2.4 is the rightmost part of the board)
xdot: the velocity of the robot (technically, this can go from -inf to inf)
theta: the angle that the pole is at (0 is straight up, -12 degrees or fewer means that the pole falls to the left, 12 degrees or more means that the pole falls to the right)
thetadot: the change in angle per second.


The robot can choose between two actions:
0: move left
1: move right

Success is balancing for 500 ticks, failure is if the stick falls more than 12 degrees from the median, or if the robot moves more than 2.4 meters from the center.

Your first task is to make a robot learn this task using reinforcement learning (Q-learning).

## OpenAI Gym:

You do not have to implement the problem domain yourself, there is a resource called openAI gym which has a set of common training examples. Gym can be installed with the following command:

	> sudo pip3 install gym

After running the provided command, you may also be asked to install some additional packages for the video encoding. You'll see an error message with instructions to follow.

### State Discretization:

We will discretize the space in order to simplify the reinforcement learning algorithm. One example can be as follows:
x: (one bucket for < -.08, one for -.08 < x < .08, one for > .08)
xdot: (one bucket for <-.5, one for -.5 < xdot < .5, one for > .5)
theta: divided into six buckets, separated by: -6deg, -1deg, 0, 1deg, 6deg
thetadot divided into three buckets, separated by: -50deg/s, 50deg/s

These state components are combined into a single integer state (0 ... 161 for this example). This is done for you in the function discretize_state in the provided code.



## Your Task (part 1):

You need to implement the q-learning part of this task (right now the template code will not do any learning). You need to implement the following equation from the lecture slides:

Q-values can be learned directly from reward feedback
	Q(a,i) ← Q(a,i) + α(R(i) +𝛾 * maxa'Q(a',j) - Q(a,i))
The magic happens (or right now does not happen) on line 112 of cart.py. In this case, the discretized state (s) would be i in this equation, and the next discretized state (sprime) would be j. Reward is stored in the variable reward, and the learning rate (α) is in the variable alpha, which is set initially in the code. The predicted value of the next state, maxa'Q(a',j), is already computed and stored in the variable predicted_value.

### Exploration vs Exploitation:
Many times the Q learning algorithm may converge to a local optima rather than global optima. To overcome this issue, an epsilon greedy strategy leveraging exploration and exploitation concept is employed such that the algorithm explores the action state space before deciding to choose a route that converges to local optima.
Exploration allows an agent to improve the current knowledge of actions resulting in policy that offers long-term benefits. 
Exploitation, on the other hand, follows the greedy policy based on the action that provides maximum Q value.
During the learning phase we want the agent to simultaneously explore as well as exploit the current knowledge it has about the environment. Here the Epsilon-greedy strategy comes into play.
Epsilon-greedy action selection: This is a simple method to balance exploration and exploitation by choosing between random actions and optimal action (based on current policy). The pseudo code is as follows:
```
p = random()
If p<e:		
	Choose random action
Else:
	Choose action that gives max Q value
```

You’ll need to implement the Epsilon-greedy strategy for choosing the action instead of choosing the action randomly(line 103 for cart.py)

You'll need to implement the equation Q-value and set the alpha and gamma values. I'd try stepping the alpha values up and down by factors of .1 in both directions and the gamma values by .01 in both directions to see the effect it has on the learning. 

The program will train over 50001 episodes and play a video of each 1000th training if the render variable is set to true.

Once your model is trained it will be saved the Q-table as ‘cart.npy’ file. Make sure that you don’t change this file name.

This is easy on the code side, but will allow you to experiment with the various factors of reinforcement learning.

## Your Task (part 2):
Now that you've implemented q-learning for one task, you will move to the mountain car task. Instead of 2 actions (left, right), this task has three (left, null, right). The task also has different state variables (only 2)

**x**: the location of the robot (-1.2 is the left, -.45 is approximately the valley, 0.6 is the rightmost part of the board, 0.5 is the location of the flag)
**xdot**: the velocity of the robot (this can go from -0.07 to 0.07)

This will require you to change the number of bins for state descritization as well as the alpha and gamma values. Additionally, you need to implement the exploration vs exploitation part for this problem as well.

Once your model is trained it will be saved the Q-table as ‘car.npy’ file. Make sure to that you don’t change this file name.

An example video of how this training can proceed is here:

https://youtu.be/81F1tE21BXA



### Instructions to join the github classroom assignment
Begin by joining the project[2] in your github classroom which will create a repository under your github account with the name project2-your_github_username . For instance if your github username is “abc”, then the name of the repository will be project-2-abc:
Assignment link for all sections:
https://classroom.github.com/a/KBLq5xA7


Now you need to clone this repository to your local system and continue working from there. With every commit you push to your remote, the github classroom setup will allow you to see how your solution fairs. There is no limit to the number of commits you can make. Also, keep in mind that there is a timeout of 1 minute for your code to run, meaning that if your code takes longer than 8 minutes to run for a single test case, it will be reported as failed.

### Turn-in Instructions:
git add <files to add>
git commit -m "<commit message>"
git push


### References for python and numpy:
http://cs231n.github.io/python-numpy-tutorial/






