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
import collections
from seagul.rl.ars.ars_zoo import ARSZooAgent, postprocess_default
from seagul.mesh import dict_to_array, act_squared, DualRewardDiv,DualRewardProd,DualRewardLin,  adim_safe_stable_nolen, mdim_safe_stable_nolen, cdim_safe_stable_nolen, mesh_dim
from seagul.zoo3_utils import do_rollout_stable,  load_zoo_agent


torch.set_default_dtype(torch.float32)

from ray import tune

def training_function(config):
    import pybullet_envs

    env_name, algo = config["env_name"], config["algo"]
    ars_iters, num_trials = config["ars_iters"], config["mdim_trials"]
    post = config["post"]
    
    if ((algo == "ddpg") and (env_name == "BipedalWalkerHardcore-v3")):
        return

    
    env, model = load_zoo_agent(env_name, algo)
    agent_folder = config['agent_folder']
    
    
    if post is not None:
        new_agent = ARSZooAgent(env_name, algo, n_workers=8, n_delta=64, postprocessor=post, step_schedule=[0.02, 0.002],  exp_schedule=[0.025, 0.0025], train_all=True, epoch_seed=True)
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
    
    for j in range(num_trials):
        o,a,r,l = do_rollout_stable(env, model)

        if type(o[0]) == collections.OrderedDict:
            o,_,_ = dict_to_array(o)

        
        o_norm = o
        #o_norm  = o_norm[200:]

        urew_arr[j] = l[0]['episode']['r']
        len_arr[j] =  l[0]['episode']['l']
        nrew_arr[j] = np.sum(r)
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



    mdim_prod = DualRewardProd(mdim_safe_stable_nolen)
    cdim_prod = DualRewardProd(cdim_safe_stable_nolen)
    adim_prod = DualRewardProd(adim_safe_stable_nolen)
    adim_div = DualRewardDiv(adim_safe_stable_nolen)

    a2_lin = DualRewardLin(act_squared, 1, -.2)
    a1_lin = DualRewardLin(act_squared, 1, -.1)
    a5_lin = DualRewardLin(act_squared, 1, -.5)
    #mdim_lin = DualRewardLin(neg_mdim, 10, 1)
    #cdim_lin = DualRewardLin(cdim_safe_stable_nolen, 10, .1)
    #adim_lin = DualRewardLin(neg_adim, 10, .5)


    
    analysis = tune.run(
        training_function,
        config={
            "ars_iters": 200,
            "mdim_trials": 10,
            "post" : tune.grid_search([None, adim_div]),
            "mdim_kwargs" : mdim_kwargs,
            "agent_folder": agent_folder,
            "env_name": tune.grid_search(["Walker2DBulletEnv-v0","HalfCheetahBulletEnv-v0","HopperBulletEnv-v0", "AntBulletEnv-v0"]),
            "algo": tune.grid_search(['a2c','ppo','ddpg','td3','sac','tqc'])
        },
        resources_per_trial= {"cpu": 8},
        verbose=2,
        fail_fast=True,
    )

    
    

    # analysis = tune.run(
    #     training_function,
    #     config={
    #         "ars_iters": 200,
    #         "mdim_trials": 10,
    #         "agent_folder": agent_folder,
    #         "env_name": tune.grid_search(["PandaReach-v1", "PandaPickAndPlace-v1", "PandaPush-v1", "PandaSlide-v1", "PandaStack-v1"]),
    #         "post" : tune.grid_search([None, postprocess_default, adim_prod, a2_lin]),#, mdim_lin, adim_lin]),
    #         "algo": tune.grid_search(['tqc'])
    #     },
    #     resources_per_trial= {"cpu": 8},
    #     verbose=2,
    #     fail_fast=True,
    # )


    # analysis = tune.run(
    #     training_function,
    #     config={
    #         "ars_iters": 200,
    #         "mdim_trials": 10,
    #         "agent_folder": agent_folder,
    #         "env_name": tune.grid_search(["FetchReach-v1", "FetchPickAndPlace-v1", "FetchPush-v1", "FetchSlide-v1"]),
    #         "post" : tune.grid_search([None, postprocess_default]),#, mdim_lin, adim_lin]),
    #         "algo": tune.grid_search(['tqc'])
    #     },
    #     resources_per_trial= {"cpu": 8},
    #     verbose=2,
    #     fail_fast=True,
    # )

    


    # analysis = tune.run(
    #     training_function,
    #     config={
    #         "ars_iters": 200,
    #         "mdim_trials": 10,
    #         "agent_folder": agent_folder,
    #         "env_name": tune.grid_search(["Pendulum-v0"]),
    #         "post" : tune.grid_search([None, postprocess_default]),
    #         "algo": tune.grid_search(['a2c', 'ppo', 'ddpg', 'sac', 'td3','tqc'])
    #     },
    #     resources_per_trial= {"cpu": 8},
    #     verbose=2,
    #     fail_fast=True,
    # )


    
    # analysis = tune.run(
    #     training_function,
    #     config={
    #         "ars_iters": 200,
    #         "mdim_trials": 10,
    #         "agent_folder": agent_folder,
    #         "env_name": tune.grid_search(["LunarLanderContinuous-v2", "BipedalWalker-v3", "BipedalWalkerHardcore-v3", "Pendulum-v0", "MountainCarContinuous-v0"]),
    #         "post" : tune.grid_search([None, postprocess_default]),
    #         "algo": tune.grid_search(['a2c', 'ppo', 'sac', 'td3','tqc', 'ddpg'])
    #     },
    #     resources_per_trial= {"cpu": 8},
    #     verbose=2,
    #     fail_fast=True,
    # )

    
    
    #"env_name": tune.grid_search(["Walker2DBulletEnv-v0","HalfCheetahBulletEnv-v0","HopperBulletEnv-v0", "AntBulletEnv-v0", "ReacherBulletEnv-v0"]),


    #"algo": tune.grid_search(['tqc'])
            
    # partial(mdim_div_stable, mdim_kwargs=mdim_kwargs),
        
    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    analysis.results_df.to_csv(csv_filename)
    print(f"saved in {csv_filename}")
