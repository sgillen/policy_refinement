# Direct Random Search for Fine Tuning of Deep Reinforcement Learning Policies

This Repo contains code to accompany the paper "Direct Random Search for Fine Tuning of Deep Reinforcement Learning Policies" submitted to RA-L/ICRA 2022

It contains one script to run the experiments used in then paper, and several notebooks to load and process data from those experiments.
Some of the code (most notably the random search implementation we use) is contained in a library of mine hosted [here](https://github.com/sgillen/seagul
). 


## Installation:

First, clone and install seagul:

```sh
git clone https://github.com/sgillen/seagul
pip install ./seagul
```

You will also need to install the rl baselines 3 zoo, to access the baseline agents

```sh
git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo
```

If you want to use the panda environments you will also need to clone the latest from here:

```sh
git clone https://github.com/qgallouedec/panda-gym
pip install ./panda_gym
```

Finally clone this repo and install the remaining requirements

```sh
git clone https://github.com/sgillen/policy_refinement/
cd policy_refinment/
pip install -r requirements.txt
```

You should now be able to train new agents, load the data used in the paper, and run baseline agents. 
To access our own agents without re running the training, see the keep_agents/ folder, which has a link to download the trained agents at.



