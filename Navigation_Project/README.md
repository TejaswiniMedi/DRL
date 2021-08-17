README

This project contains an agent based on Deep Reinforcement Learning that can learn from zero (any labeled data) to 
collect yellow bananas instead of blue bananas in a vast, square world.
The Goal of project is as follows:
In the project Navigation a unity environment is given, in which yellow and blue bananas are placed. 
These bananas can be collected, for every yellow banana a positive reward of +1 is given, for every blue banana a negative reward of -1. 
The goal is to create an agent which can get a reward of +13 over 100 consecutive episodes.
The agent can navigate in the environment in the following discrete action space:
0 moves forward
1 moves backward
2 turns left
3 turns right
The state space has 37 dimensions containing the agent's velocity and ray-based perception of objects around the forward direction of the agent.

The following steps are followed to acheive the target of the project
1.Set-up the Environment
2.Start the Environment
3.Examine the state and Action Spaces
4.Train the Agent 
5.Save the weights of model
6.Load the weights 
7.Test the agent on environment and close the environment.

Files Description:

DQN.py is the code containing the Q-Network used as the function approximator by the agent.
Agent.py is the code for the agent used in environment.
model.pth is saved model weights for the original DQN model.
Navigation.ipynb is notebook containing solution 

Installation:
The installation process is divided in three parts:
1.Python 3.6.x
2.Dependencies
3.Unity's Environment

Dependencies
Dependencies are downloaded by using pip commands 
example: !pip install dependency_name
Use the requirements.txt file to install the required dependencies via pip.
pip install -r requirements.txt


Select the environment that matches our operating system
NOTE:Information about the unity enrionment and the operating system is given in udacity-deep reinforcement learning repository in github
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out (https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) 
	if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.



