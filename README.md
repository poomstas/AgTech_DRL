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

![Figure]('https://github.com/poomstas/AgTech_DRL/blob/main/README_Figures/A.png')

![Fig]('https://github.com/poomstas/AgTech_DRL/blob/main/README_Figures/A.png?raw=true')

![Fig1](https://github.com/poomstas/AgTech_DRL/main/README_Figures/A.png?raw=true)

![Fig2]('https://github.com/poomstas/AgTech_DRL/blob/main/README_Figures/A.png?raw=true')

![Fig3]('README_Figures/A.png?raw=true')

![Fig4]('README_Figures/A.png')

![Figure]('.blob/main/README_Figures/A.png')

![Dali]('https://i0.wp.com/rayhaber.com/wp-content/uploads/2021/01/salvador-dali-kimdir.jpg)

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
