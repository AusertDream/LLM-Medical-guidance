import os
from time import sleep

import wandb
import random

wandb.init(project='test',
           settings=wandb.Settings(start_method='thread', console='off'),
           name="sleep",
           config={
               'learning_rate': 0.01,
               'architecture': 'CNN',
               'dataset': 'random',
               'batch_size': 32,
               "epoch": 100
               })

if __name__ == '__main__':

    wandb.finish()