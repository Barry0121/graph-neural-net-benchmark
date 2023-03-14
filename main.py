from train import *

from src.train import *
from src.models.args import *
from src.visualization import *

# initalize argments
# you can change parameters in `src/models/args.py`
args = Args()

print("Start Training...")
# this will train the models
# saved models and weights will be at the specified path in args.py
train(args, train_inverter=True)
print("Finish Training!")
get_tsne_plots(args)
get_degree_distribution_plots(args)
plot_losses()
print("Plot generated, check src/visualizations for grpahs.")