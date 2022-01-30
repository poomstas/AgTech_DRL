# Applying Deep Reinforcement Learning to Agricultural Tech
## Problem Statement and Overview

The objective of this project is to maximize the the expected crop yield by optimizing over 13 continuous control (action) variables. The expected crop yield is determined using the PCSE-v0 crop simulator provided in https://github.com/poomstas/spwk-agtech-task.git.

The 

The problem is episodic, and it runs until the growth

The problem is episodic: each episode begins as the wheat is planted, and ends when the 

The task to be completed involves a crop cultivation simulation model provided as a Python gym environment. 

Further information regarding the model is provided. The model follows a markov decision process (MDP) framework, with 11 observation (state) variables, and 13 continuous action variables. Further details regarding these variables are provided in the table below. 

**Table: Observation Variables**

Variable Name	|	Variable	|	Min	|	Max	|	Unit	|
---	|	---	|	---	|	---	|	---	|
DVS	|	Development Stage	|	0	|	2	|	Stage	|
LAI	|	Leaf Area Index	|	0	|	10	|	ha/ha	|
TAGP	|	Total Above Ground Production	|	105	|	30,000	|	kg/ha	|
TWSO	|	Total Dry Weight of Storage Organs	|	0	|	11,000	|	kg/ha	|
TWLV	|	Total Dry Weight of Leaves	|	68.25	|	7,500	|	kg/ha	|
TWST	|	Total Dry Weight of Stems	|	36.75	|	12,500	|	kg/ha	|
TWRT	|	Total Dry Weight of Roots	|	105	|	4,500	|	kg/ha	|
TRA	|	Crop Transpiration Rate	|	0	|	2	|	cm/day	|
RD	|	Rooting Depth	|	10	|	120	|	cm	|
SM	|	Soil Moisture	|	0.3	|	0.57	|	cm<sup>3</sup>/cm<sup>3</sup>	|
WWLOW	|	Total Amnt of Water in the Soil Profile	|	54.177	|	68.5	|	cm	|


**Table: Control Variables (Continuous)**

Variable Name	|	Variable	|	Min	|	Max	|	Unit	|
---	|	---	|	---	|	---	|	---	|
IRRAD	|	Incoming Global Radiation	|	0	|	4.0 × 10<sup>7</sup>	|	J/m<sup>2</sup>/day	|
TMIN	|	Daily Min Temp	|	-50	|	60	|	Celsius	|
TMAX	|	Daily Max Temp	|	-50	|	60	|	Celsius	|
VAP	|	Daily Mean Vapor Pressure	|	0.06 × 10<sup>-5</sup>	|	199.3 × 10<sup>-4</sup>	|	hPa	|
RAIN	|	Daily Total Rainfall	|	0	|	25	|	cm/day	|
E0	|	Penman Potential Evaporation from a Free Water Surface	|	0	|	2.5	|	cm/day	|
ES0	|	Penman Potential Evaporation from Moist Bare Soil Surface	|	0	|	2.5	|	cm/day	|
ET0	|	Penman or Penman-Monteith Potential Evaporation for a Reference Crop Canopy	|	0	|	2.5	|	cm/day	|
WIND	|	Daily Mean Wind Speed at 2m Height	|	0	|	100	|	m/sec	|
IRRIGATE	|	Amnt of Irrigation in cm water applied on this day	|	0	|	50	|	cm	|
N	|	Amnt of N fertilizer in kg/ha applied on this day	|	0	|	100	|	kg/ha	|
P	|	Amnt of P fertilizer in kg/ha applied on this day	|	0	|	100	|	kg/ha	|
K	|	Amnt of K fertilizer in kg/ha applied on this day	|	0	|	100	|	kg/ha	|


In the given gym environment, the above action variables are scaled from -1 to 1 using the minimum and maximum values for computational convenience.

As outlined in the GitHub problem description page, the ultimate objective of the task is to create an agent that 1) maximizes the net profit, 2) maintains high training stability, and 3) achieves fast convergence.

The report continues with the Executive Summary section whereI make the final recommendation. Then the three attempts are summarized in the order they were conducted, followed by the Conclusion and Future Works sections at the end.

---

This repo contains an application of the a few deep reinforcement learning techniques to a plant growth simulator to optimize the growth conditions to maximize plant productivity.

This work uses a crop cultivation simulation model provided as a python gym environment. 

The crop to be studied is wheat, and the variables involved can be largely separated into two groups: observation variables, and action variables. 

The model follows a markov decision process (MDP) frameowrk, with 11 observation (state) variables, and 13 continuous action variables. Further details regarding these variables are provided in the table below. 

---

## Executive Summary
Three reinforcement learning techniques were used to maximize the total episodic reward: DDPG, TD3 and SAC. I have selected these methods primarily because they:

1. can be applied to problems wherein the action variables are continuous,

2. achieve high sampling efficiency through replay buffers, and

3. have proved to be successful (i.e. high stability & fast convergence) in other similar environments.

Further discussions on the inner workings and the implementational details of the algorithms are provided in the respective sections of the report.

As summarized in the table below, the SAC yielded the best results, and was selected to be the final submission for the challenge. This section summarizes the performance of the trained SAC model.
\
\
**Table: Max Total Episodic Reward for Three Algorithms**

DRL Algorithm	|	Max Total Episodic Reward ($/ha)	|
---	|	---	|
DDPG	|	0	|
TD3	|	1,406	|
SAC	|	2,802	|
\
Because the SAC algorithm is inherently stochastic, it gives different results every time it is run. To evaluate the trained model accurately, I executed the SAC algorithm 1,000 times to visualize the total reward distribution. The resulting graph and data are provided below:
\
\
**Table: Average and Maximum Total Episodic Reward for SAC**

Average Total Episodic Reward ($/ha) |	Max Total Episodic Reward ($/ha)|
---	|	---	|
2250.47	|	2802.47	|

**Figure: Distribution of Total Episodic Rewards Retrieved from 1,000 Episodes**
<p align="center">
  <img src="/README_Figures/A.png" width="450" title="SAC Final Model Performance">
</p>

The hyperparameters used to train the SAC model is summarized in the table below:

**Table: Hyperparameters Used to Train the Final Selected Model (SAC)**

Hyperparameter	|	Value	|
---	|	---	|
Optimizer	|	Adam	|
alpha (learning rate for actor)	|	0.001	|
beta (learning rate for critic)	|	0.001	|
Discount Factor	|	0.99	|
tau	|	0.01	|
Reward Scale	|	18	|
Batch Size	|	100	|
Replay Buffer Size	|	106	|
Layer 1 Size	|	256	|
Layer 2 Size	|	256	|
Max Timesteps Per Episode	|	50,000	|

The subsequent sections details the attempts in the order they were made.

# Trial #1: Deep Deterministic Policy Gradient (DDPG)
## Algorithm

The first algorithm used was Deep Deterministic Policy Gradient (DDPG). DDPG is a deep reinforcement learning technique that draws from both Q-learning and policy gradients. One of the motives for creating DDPG was that Deep Q-Network (DQN) could only handle cases where the action spaces were discrete and low-dimensional. This was the primary basis for selecting DDPG as my first attempt to solve the problem, which involves a continuous action variable.

DDPG learns a Q-function and a policy simultaneously, and uses off-policy data and the Bellman equation to modify the Q-function. It then uses the Q-function to update the policy (Lillicrap et al., 2016). Simply put, DDPG is an approach that attempts to solve one of the major limitations of DQN (i.e. the requirement that the action space is discrete and low-dimensional). DDPG simultaneously draws from the successes of DQN by implementing two of its ideas: the replay buffer and the target network.

DDPG overcomes the above limitation by taking advantage of the fact that when the action space is continuous, the (optimal) action-value function is differentiable with respect to the action variable. Using this, a gradient-based learning rule for a policy can be constructed, as below.

<p align="center">
  <img src="/README_Figures/AB.png" width="800" title="DDPG Main Equation">
</p>

The gradient values are then used to update the Q-function and the policy. Here, soft-updating is used to ensure that the updating procedure retains some stability.

The overview of the DDPG algorithm in the form of pseudocode is provided below.

<p align="center">
  <img src="/README_Figures/B.png" width="800" title="DDPG Main Equation">
</p>

# Running the Scripts
## Create a Conda Environment
```
conda create --name spacewalk_test python=3.8
conda activate spacewalk_test
pip install git+https://github.com/poomstas/spwk-agtech-task.git
python -m spwk_agtech.make_weather_cache
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge tensorboard
conda install matplotlib
```

## Running the Training Scripts
### The main training scripts in each folder begin with `main`. 

`main_train_ddpg.py` for DDPG

`main_train_td3.py` for TD3

`main_train_sac.py` for SAC


### Specifying Hyperparameters
When running the training scripts, hyperparameters can be specified. For instance:

`python main_train_ddpg.py --alpha 0.01 --beta 0.1 --tau 0.01 --gamma 0.95 --batch_size 32 --layer1_size 300 --layer2_size 200`

To see which hyperparameters can be specified, run: 

`python main_train_ddpg.py --help`


## Check Best-Performing Action Set
To visualize the results of the best-performing action set (acquired by training an SAC model), run:

`python check_best_performing_action.py`


## Load the Best Trained SAC Model and Test
To load the best-case SAC model, run multiple episodes on the given environment and calculate the average and maximum episode rewards,
run the following:

`python test_SAC_model.py`

You can adjust the total number of episodes to run in the script by adjusting the `N_TEST_CASE` variable in the script.
