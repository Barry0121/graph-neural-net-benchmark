"""
Run this file to train WGAN model
"""

from src.train import *
from src.models.args import *
from src.visualizations import *

# initalize argments
# you can change parameters in `src/models/args.py`
args = Args()
# this will train the models
# saved models and weights will be at the specified path in args.py
train(args, train_inverter=True)

# TODO: add viz generation script
