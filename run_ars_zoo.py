import pybullet_envs
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import xarray as xr
import matplotlib.pyplot as plt
import copy
import pybullet_envs
import pandas as pd

from seagul.rl.ars.ars_zoo import ARSZooAgent
from seagul.mesh import mdim_div_stable, mesh_dim
from seagul.rollouts import do_rollout_stable
from seagul.rollouts import load_zoo_agent


torch.set_default_dtype(torch.float32)

from ray import tune

def training_function(config):
    import pybullet_envs

    train, env_name, algo, ars_iters, num_trials = config["train"], config["env_name"], config["algo"], config["ars_iters"], config["mdim_trials"]
    env, model = load_zoo_agent(env_name, algo)
    
    if train:
        new_agent = ARSZooAgent(env_name, algo, n_workers=12, n_delta=64, postprocessor=mdim_div_stable, step_schedule=[0.05, 0.005],  exp_schedule=[0.05, 0.005])
        new_agent.learn(ars_iters, verbose=False) 


    for i in range(num_trials):
        o,a,r,l = do_rollout_stable(env, model)
        o_norm = env.normalize_obs(o).squeeze()
        o_norm  = o_norm[200:]
        rew = np.sum(r)
    
        try:
            mdim, _, _, _ = mesh_dim(o_norm)
        except:
            mdim = np.nan

    tune.report(mdim=mdim.mean(), reward=rew.mean())


analysis = tune.run(
    training_function,
    config={
        "ars_iters": 100,
        "mdim_trials": 10,
        "train" : tune.grid_search([False, True]),
        "env_name": tune.grid_search(["Walker2DBulletEnv-v0","HalfCheetahBulletEnv-v0","HopperBulletEnv-v0"]),
        "algo": tune.grid_search(['ppo', 'a2c'])
    },
    resources_per_trial= {"cpu": 12},
    verbose=1,
)

# Get a dataframe for analyzing trial results.
df = analysis.results_df
analysis.results_df.to_csv('ppo_a2c_100.csv')
