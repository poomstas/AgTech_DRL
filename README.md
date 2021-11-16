# Spacewalk Interview Assignment Submission 

This repo is a submission for the AgTech interview assignment at Spacewalk.

## Create a Conda Environment
```
conda create --name spacewalk_test python=3.8
conda activate spacewalk_test
pip install git+https://github.com/TeamSPWK/spwk-agtech-task.git
python -m spwk_agtech.make_weather_cache
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge tensorboard
conda install matplotlib
```

## Running the Training Scripts
The main training scripts in each folder begin with `main`. 

`main_train_ddpg.py` for DDPG

`main_train_td3.py` for TD3

`main_train_sac.py` for SAC


## Specifying Hyperparameters for Training
Hyperparameters can be specified as follows (example):

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
