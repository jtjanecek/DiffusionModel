# DiffusionModel

A python implementation for easily running JAGS bayesian group comparison models that are based on the Yassa Young & Old Diffusion paper.

## Installation
If you are coming from the [DiffusionModelContainer](https://github.com/jtjanecek/DiffusionModelContainer) repo, this will be installed automatically in the Singularity Image.
```
git clone https://github.com/jtjanecek/DiffusionModel.git
cd DiffusionModel
pip install .
```

## Code Documentation
For a full example, see [here](https://github.com/jtjanecek/DiffusionModel/blob/master/example/code/run.py) 

### RawData
This will be used for converting your CSVs into Pandas dataframes and eventually numpy arrays. You need to pass in a groups csv, reaction time csv, and condition csv (with headers). You can see more details [here](https://github.com/jtjanecek/DiffusionModelContainer/blob/master/README.md#setting-up-your-data) on the csv input format. [Example data](https://github.com/jtjanecek/DiffusionModel/tree/master/example/data).
```
from diffusion_model.raw_data import RawData
data = RawData(
        groups_file="/data/groups.csv",
        rts_file="/data/rt.csv",
        conds_file="/data/conds.csv"
)

```

### Group Comparison Models
These models will be the same that were used in the Young vs Old study. `GroupComparison().build_models(data)` will construct the models, monitor params, and initialization values.
```
models = GroupComparison().build_models(data)
```
`models` here will be a list of dictionaries representing each model. Each dict will have a jags model, inits, and monitor params.


### Diffusion Model methods
Methods on the DiffusionModel class
```
###############################################
initialize(
	model_name=None,
	raw_data=None,
	jags_model=None,
	nchain=6, 
	nburn=100, 
	niter=200, 
	nthin=1, 
	inits={},
	monitor_params=[])


###############################################
R


###############################################
execute(model_name, chain='all', parallel=True)
###############################################
Run JAGS on the specified chains.

params:
	model_name: The name of the model. Must have a folder generated from initialization


dm.execute('youngold_model_00', chain='all', parallel=True)
dm.format_chains('youngold_model_00', niter=1000, nthin=1, chain='all')
dm.combine_chains('youngold_model_00')
dm.analyze_convergence('youngold_model_00')
```
