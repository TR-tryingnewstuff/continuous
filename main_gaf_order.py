#%%
from typing import Dict
import numpy as np
import pandas as pd
from get_data_stock_trading_mbfa import *
from helpers_mbfa import *

import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import ray
from ray.rllib.algorithms import ppo
from ray import tune
from ray import air

from custom_model_gaf import  Continuous_CNN
from ray.rllib.models.catalog import ModelCatalog
from pyts.image import GramianAngularField
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image
from market_class import Market_continuous


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


toml = get_toml_data('config.toml')


# --------------------------------------------
    

ModelCatalog.register_custom_model("keras_model", Continuous_CNN)

config = ppo.PPOConfig().environment(Market_continuous)
config = config.rollouts(num_rollout_workers=1).resources(num_cpus_for_local_worker=3)
config = config.training(lr_schedule=toml['model']['lr_schedule'],clip_param=0.25, gamma=0.95, use_critic=True, use_gae=True, model={'custom_model': 'keras_model'}, train_batch_size=400)


tuner = tune.Tuner(  
        "PPO",
        param_space=config.to_dict(),
       # tune_config=tune.TuneConfig(max_concurrent_trials=1),
        run_config=air.RunConfig(
                    stop={"training_iteration": 100},#toml['model']['training_iteration']},
                    checkpoint_config=air.CheckpointConfig(num_to_keep= 2,checkpoint_at_end=True, checkpoint_frequency=50)

        )
)

results = tuner.fit()

# ---------------------- GET BEST ALGO FILE ---------------------------

best_result = results.get_best_result(metric="episode_len_mean", mode="max")

checkpoint_path = best_result.checkpoint
print('\n'*5, checkpoint_path, '\n'*5)

