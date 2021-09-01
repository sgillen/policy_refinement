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
from seagul.mesh import mdim_div_stable, mesh_dim, mesh_find_target_d, mdim_div_stable_nolen, cdim_div_stable_nolen, mdim_div_panda, cdim_div_panda
from seagul.zoo3_utils import do_rollout_stable,  load_zoo_agent


torch.set_default_dtype(torch.float32)

from ray import tune

def training_function(config):
    import pybullet_envs

    env_name, algo = config["env_name"], config["algo"]
    ars_iters, num_trials = config["ars_iters"], config["mdim_trials"]
    post = config["post"]
    env, model = load_zoo_agent(env_name, algo)
    agent_folder = config['agent_folder']

    
    if post is not None:

        new_agent = ARSZooAgent(env_name, algo, n_workers=8, n_delta=64, postprocessor=post, step_schedule=[0.05, 0.005],  exp_schedule=[0.05, 0.005])
        new_agent.learn(ars_iters, verbose=True)
        model = new_agent.model

        post_name = post.__name__
        print(post_name)

        os.makedirs(f"{agent_folder}/{algo}/{env_name}", exist_ok=True)
        model.save(f"{agent_folder}/{algo}/{env_name}/{post_name}.pkl")


        
    else:
        post_name = 'iden'

    mdim_arr = np.zeros(num_trials)
    cdim_arr = np.zeros(num_trials)

    nrew_arr = np.zeros(num_trials)
    urew_arr = np.zeros(num_trials)
    len_arr = np.zeros(num_trials)
    rew_arr = np.zeros(num_trials)
    
    for j in range(num_trials):
        odict,a,r,l = do_rollout_stable(env, model)
        
        o_list = []
        ach_list = []
        des_list = []
        for thing in odict:
            o_list.append(thing['observation'])
            ach_list.append(thing['achieved_goal'])
            des_list.append(thing['desired_goal'])
    
        o = np.stack(o_list).squeeze()
        ach = np.stack(ach_list).squeeze()
        des = np.stack(des_list).squeeze()

        
        o_norm = o
        #o_norm  = o_norm[200:]
        rew_arr[j] = np.sum(r)
        try:
            mdim_arr[j], cdim_arr[j], _, _ = mesh_dim(o_norm)
        except:
            mdim_arr[j] = np.nan
            cdim_arr[j] = np.nan

            

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



    # analysis = tune.run(
    #     training_function,
    #     config={
    #         "ars_iters": 100,
    #         "mdim_trials": 10,
    #         "post" : tune.grid_search([None, mdim_div_stable_nolen, cdim_div_stable_nolen]),
    #         "mdim_kwargs" : mdim_kwargs,
    #         "agent_folder": agent_folder,
    #         "env_name": tune.grid_search(["Walker2DBulletEnv-v0","HalfCheetahBulletEnv-v0","HopperBulletEnv-v0", "AntBulletEnv-v0", "ReacherBulletEnv-v0"]),
    #         "algo": tune.grid_search(['ppo', 'td3', 'sac', 'tqc'])
    #     },
    #     resources_per_trial= {"cpu": 8},
    #     verbose=2,
    #     fail_fast=True,
    # )


    analysis = tune.run(
        training_function,
        config={
            "ars_iters": 100,
            "mdim_trials": 10,
            "post" : tune.grid_search([None, mdim_div_panda, cdim_div_panda]),
            "agent_folder": agent_folder,
            "env_name": tune.grid_search(["PandaReach-v1", "PandaPickAndPlace-v1", "PandaPush-v1", "PandaSlide-v1", "PandaStack-v1", "FetchReach-v1", "FetchPickAndPlace-v1", "FetchPush-v1", "FetchSlide-v1"]),
            "algo": tune.grid_search(['tqc'])
        },
        resources_per_trial= {"cpu": 8},
        verbose=2,
        fail_fast=True,
    )

    
    # partial(mdim_div_stable, mdim_kwargs=mdim_kwargs),
        
    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    analysis.results_df.to_csv(csv_filename)
