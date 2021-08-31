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
from seagul.mesh import mdim_div_stable, mesh_dim, mesh_find_target_d, mdim_div_stable_nolen, cdim_div_stable_nolen
from seagul.zoo3_utils import do_rollout_stable,  load_zoo_agent


torch.set_default_dtype(torch.float32)

from ray import tune

def training_function(config):
    import pybullet_envs

    env_name, algo = config["env_name"], config["algo"]
    ars_iters, num_trials = config["ars_iters"], config["mdim_trials"]
    mdim_kwargs, post = config["mdim_kwargs"], config["post"]
    env, model = load_zoo_agent(env_name, algo)
    agent_folder = config['agent_folder']

    
    if post is not None:

        new_agent = ARSZooAgent(env_name, algo, n_workers=8, n_delta=64, postprocessor=post, step_schedule=[0.05, 0.005],  exp_schedule=[0.05, 0.005])
        new_agent.learn(ars_iters, verbose=True)
        model = new_agent.model

        post_name = post.__name__
        print(post_name)
        os.makedirs(f"{agent_folder}/{env_name}", exist_ok=True)
        model.save(f"{agent_folder}/{env_name}/{post_name}.pkl")

        print(os.path.dirname(os.path.realpath(__file__)))
        
    else:
        post_name = 'iden'

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

        #o_mdim  = o[200:]
        o_mdim = o
        try:
            mdim_arr[i], cdim_arr[i], _, _ = mesh_dim(o_mdim, **mdim_kwargs)
        except:
            mdim_arr[i] = np.nan
            cdim_arr[i] = np.nan

            

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
                post = post_name,
                len_mean = len_arr.mean())



if __name__ == "__main__":

    csv_folder_name = "test_out"
    agent_folder_name = "test_agent"
    root_folder = os.path.dirname(os.path.realpath(__file__))

    output_name = input("Output file name: ")

    csv_filename = f"{root_folder}/{csv_folder_name}/{output_name}.csv"
    agent_folder = f"{root_folder}/{agent_folder_name}/{output_name}/"

    print(os.getcwd())
    input(f"will save csv in {csv_filename}, agents in {agent_folder}    ok?") 
    if os.path.exists(csv_filename):
        input(f"Filename  already exists, overwrite?")

    mdim_kwargs= {"upper_size_ratio":1.0, "lower_size_ratio":0.0}

    

    # analysis = tune.run(
    #     training_function,
    #     config={
    #         "ars_iters": 100,
    #         "mdim_trials": 10,
    #         "post" : tune.grid_search([None, mdim_div_stable_nolen, cdim_div_stable_nolen]),
    #         "mdim_kwargs" : mdim_kwargs,
    #         "agent_folder": agent_folder,
    #         "env_name": tune.grid_search(["Walker2DBulletEnv-v0", "HalfCheetahBulletEnv-v0", "W"]),
    #         "algo": tune.grid_search(['ppo'])
    #     },
    #     resources_per_trial= {"cpu": 24},
    #     verbose=2,
    #     fail_fast=True,
    # )



    analysis = tune.run(
        training_function,
        config={
            "ars_iters": 100,
            "mdim_trials": 10,
            "post" : tune.grid_search([None, mdim_div_stable_nolen, cdim_div_stable_nolen]),
            "mdim_kwargs" : mdim_kwargs,
            "agent_folder": agent_folder,
            "env_name": tune.grid_search(["Walker2DBulletEnv-v0","HalfCheetahBulletEnv-v0","HopperBulletEnv-v0", "AntBulletEnv-v0", "ReacherBulletEnv-v0"]),
            "algo": tune.grid_search(['ppo', 'td3', 'sac', 'tqc'])
        },
        resources_per_trial= {"cpu": 8},
        verbose=2,
        fail_fast=True,
    )

    # partial(mdim_div_stable, mdim_kwargs=mdim_kwargs),
        
    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    analysis.results_df.to_csv(csv_filename)
