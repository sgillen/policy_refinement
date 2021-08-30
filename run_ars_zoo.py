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
from functools import partial

from seagul.rl.ars.ars_zoo import ARSZooAgent
from seagul.mesh import mdim_div_stable, mesh_dim, mesh_find_target_d
from seagul.rollouts import do_rollout_stable
from seagul.rollouts import load_zoo_agent


torch.set_default_dtype(torch.float32)

from ray import tune

def training_function(config):
    import pybullet_envs

    train, env_name, algo = config["train"], config["env_name"], config["algo"]
    ars_iters, num_trials, mdim_kwargs = config["ars_iters"], config["mdim_trials"], config["mdim_kwargs"]
    env, model = load_zoo_agent(env_name, algo)

    
    if train:
        mdim_post = partial(mdim_div_stable, mdim_kwargs=mdim_kwargs)
        new_agent = ARSZooAgent(env_name, algo, n_workers=8, n_delta=64, postprocessor=mdim_post, step_schedule=[0.025, 0.0025],  exp_schedule=[0.025, 0.0025])
        new_agent.learn(ars_iters, verbose=False)
        model = new_agent.model

    mdim_arr = np.zeros(num_trials)
    cdim_arr = np.zeros(num_trials)

    nrew_arr = np.zeros(num_trials)
    urew_arr = np.zeros(num_trials)
    len_arr = np.zeros(num_trials)
    rew_arr = np.zeros(num_trials)
    dcrit_arr = np.zeros(num_trials)
    
    for i in range(num_trials):
        o,a,r,info = do_rollout_stable(env, model)
        nrew_arr[i] = np.sum(r)
        urew_arr[i] = info[0]['episode']['r']
        len_arr[i] =  info[0]['episode']['l']

        o_mdim  = o[200:]
        try:
            mdim_arr[i], cdim_arr[i], _, _ = mesh_dim(o_mdim, **mdim_kwargs)
        except:
            mdim_arr[i] = np.nan
            cdim_arr[i] = np.nan


        # try:
        #     dcrit_arr[i] = mesh_find_target_d(o_mdim)
        # except:
        #     dcrit_arr[i] = np.nan
            

    tune.report(mdim_mean =     mdim_arr.mean(),
                mdim_std =      mdim_arr.std(),
                mdim_nan_mean = np.nanmean(mdim_arr),
                mdim_nan_std =  np.nanstd(mdim_arr),
                cdim_mean =     cdim_arr.mean(),
                cdim_std =      cdim_arr.std(),
                cdim_nan_mean = np.nanmean(cdim_arr),
                cdim_nan_std =  np.nanstd(cdim_arr),
                nreward_mean =  nrew_arr.mean(),
                nreward_std =   nrew_arr.std(),
                ureward_mean =  urew_arr.mean(),
                ureward_std =   urew_arr.std(),
                # dcrit_mean =    dcrit_arr.mean(),
                # dcrit_std =     dcrit_arr.std(),
                len_mean = len_arr.mean())



if __name__ == "__main__":
    csv_name = input("Output file name: ")

    if os.path.exists(csv_name):
        input(f"Filename {csv_name} already exists, overwrite?")


    analysis = tune.run(
        training_function,
        config={
            "ars_iters": 100,
            "mdim_trials": 10,
            "mdim_kwargs": {"upper_size_ratio":1.0, "lower_size_ratio":0.0},
            "train" : tune.grid_search([False]),
            "env_name": tune.grid_search(["Walker2DBulletEnv-v0","HalfCheetahBulletEnv-v0","HopperBulletEnv-v0", "AntBulletEnv-v0", "ReacherBulletEnv-v0"]),
            "algo": tune.grid_search(['ppo', 'a2c', 'td3', 'sac','ddpg', 'tqc'])
        },
        resources_per_trial= {"cpu": 8},
        verbose=2,
    )
        
        
    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    analysis.results_df.to_csv(csv_name)
