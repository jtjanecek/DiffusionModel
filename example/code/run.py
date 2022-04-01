from diffusion_model.diffusion_model import DiffusionModel
from diffusion_model.raw_data import RawData
from diffusion_model.group_comparison import GroupComparison
import time

# Initialize the raw data. These paths are the paths inside the container
data = RawData(
        groups_file="/data/groups.csv",
        rts_file="/data/rt.csv",
        conds_file="/data/conds.csv"
)

# Initialize the group comparison models 
models = GroupComparison().build_models(data)

# Set the workdir. This is the path inside the container
dm = DiffusionModel(work_dir="/workdir")

# models_to_run is a list of all models. You can run them all at once, or
# (preferred) run them separately and in parallel. models_to_run[5] is
# the 6th model (there are 8 total)
model = models[1]

# Initalize this model. This will create the model folder in workdir
# if it doesn't exist. And set things up for execution
# This takes less than 10 seconds
dm.initialize(
        model_name=model['model_name'],
        raw_data=data,
        jags_model=model['jags_model'],
        nchain=6,
        nburn=100,
        niter=500,
        nthin=1,
        inits=model['inits'], 
        monitor_params=model['params']
)

# This will execute the simulation of the chains. This can take 
# upwards of 18+ hours depending on the simulation parameters
# (nburn, niter, nthin). This can also take upwards of 200 GB 
# of main memory
dm.execute(model['model_name'])

# If you are saving your data onto a networked drive, it can take some
# time for it to finish writing to the network disk. So this sleep 
# waits some time before reading the final results, as it can error
time.sleep(3600)

# At this point, JAGS has written a TON of text files to your model's
# directory in your workdir. These are plain text files that need
# to be formatted into matrices to be analyzed. This step formats
# them into matrices, and saves them in hdf5 format
dm.format_chains(model['model_name'])

# This step will combine the previous chain matrices into one single
# file that can more easily be used
dm.combine_chains(model['model_name'])

# This step will execute the Rhat identity method of analyze convergence
# of the chains. If any chains have >1.05 Rhat, the traceplot will be
# graphed automatically in the workdir/model_dir/convergence_plots/ 
# directory. It will also print to the console saying that "Rhat > 1.05"
dm.analyze_convergence(model['model_name'])
