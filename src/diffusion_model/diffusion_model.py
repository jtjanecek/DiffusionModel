from copy import deepcopy
import logging
logger = logging.getLogger("diffusion_model")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)s | %(levelname)s | %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)
from time import sleep
import pandas as pd
import os
import glob
import random
from decimal import Decimal
import numpy as np
import multiprocessing as mp
from arviz import rhat as gelman_rubin
from datetime import datetime
import h5py
import gc
from itertools import product
import importlib.resources as pkg_resources
import matplotlib.pyplot as plt
import json
try:
	from . import templates
	from .templates.inits import InitialValues
	from .chain import Chain
except ImportError:
	logger.error("Failed to import templates")
	import templates
	from templates.inits import InitialValues
	from chain import Chain

class DiffusionModel:
	def __init__(self, work_dir=None):
		if work_dir == None:
			raise Exception('You need to specify a working directory for the model')

		if not os.path.isdir(work_dir):
			logger.info(f"Working directory ({work_dir}) does not exist. Creating it for you.")	
			os.makedirs(work_dir)

		self._work_dir = os.path.abspath(work_dir)
		logger.info(f"Using work dir: {self._work_dir}")


	def format_chains(self, model_name, chain='all'):
		model_dir = os.path.join(self._work_dir, model_name)
		os.chdir(model_dir)


		chains = glob.glob(f'{model_name}_chain?.script')
		if chain == 'all':
			files = []
			base_strs = [s.split("_index")[0] for s in glob.glob(f'{model_name}_chain?_index.txt')]
			for base in base_strs:
				chain_file = f'{base}_chain1.txt'
				index_file = f'{base}_index.txt'
				model_info = self.__read_model_info(f'{base}.info')
				Chain(model_info['niter'], model_info['nthin'], chain_file, index_file, '.')
		else:
			model_info = self.__read_model_info(f'{model_name}_chain{chain}.info')
			chain_file = f'{model_name}_chain{chain}_chain1.txt'
			index_file = f'{model_name}_chain{chain}_index.txt'
			Chain(model_info['niter'], model_info['nthin'], chain_file, index_file, '.')
		sleep(60)

	def __read_model_info(self, info_file):
		with open(info_file, 'r') as f:
			data = f.read().strip()
			return json.loads(data)

	def execute(self, model_name, chain='all', parallel=True):
		start_time = datetime.now()

		model_dir = os.path.join(self._work_dir, model_name)

		if chain == 'all':
			logger.info("Executing all chains ...")
			# detect all chains
			chains = glob.glob(os.path.join(model_dir,f'{model_name}_chain?.script'))
			if len(chains) == 0:
				raise Exception(f'No chains found for model {model_name}')
			if parallel == True:
				processes = []
				for chain_name in chains:
					p = mp.Process(target=self._run_chain, args=(chain_name, model_dir))
					p.start()
				processes.append(p)
				for p in processes:
					p.join()
			else:
				for chain_name in chains:
					self._run_chain(chain_name, model_dir)
		else:
			chain_name = f'{model_name}_chain{chain}.script'
			self._run_chain(chain_name, model_dir)
			
		logger.info(f"Done executing. Execution time: {str(datetime.now() - start_time)}")
		sleep(120) # Sleep 120 seconds to allow some time for the files to write

	def _run_chain(self, chain, model_dir):
		os.chdir(model_dir)
		if not os.path.isfile(chain):
			raise Exception(f'Model script: {chain} does not exist! (Is the model_dir: {self.model_dir} correct?)')
		process_code = os.system(f'jags {chain}')
		if process_code != 0:
			raise Exception('Error executing chain!')


	def initialize(self, 
			model_name=None,
			raw_data=None,
			jags_model=None,
			nchain=6, 
			nburn=100, 
			niter=200, 
			nthin=1, 
			inits={},
			monitor_params=[]
			):
		if model_name == None:
			raise Exception('No model name specified')
		logger.info(f"Initializing {model_name} ...")

		model_dir = os.path.join(self._work_dir, model_name)
		if not os.path.isdir(model_dir):
			os.makedirs(model_dir)

		os.chdir(model_dir)

		for chain in range(1,nchain+1):
			model_chain_name = f'{model_name}_chain{chain}'	
			logger.debug(f"Writing files for {model_chain_name}")

			logger.debug("Writing input data to work dir ...")
			self._write_input_data(model_chain_name, raw_data.groups, raw_data.rts, raw_data.conds)

			logger.debug("Writing seed data to work dir ...")
			self._write_seed_data(model_chain_name)

			logger.debug("Writing init data to work dir ...")
			self._write_init_data(model_chain_name, inits)
			
			logger.debug(f"Formatting {model_chain_name} ...")
			
			with open(jags_model, 'r') as f:
				model_file = f.read()
			self._write_model_file(model_chain_name, model_file, raw_data.rts)

			self._write_script(model_chain_name, nburn, niter, nthin, monitor_params)

			self._write_basic_info(model_chain_name, nchain, nburn, niter, nthin)

	def _write_basic_info(self, model_chain_name, nchain, nburn, niter, nthin):
		with open(f'{model_chain_name}.info', 'w') as data_of:	
			data_of.write(json.dumps({
					'nchain': nchain,
					'nburn': nburn,
					'niter': niter,
					'nthin': nthin
				}))

	def _write_input_data(self, model_chain_name, groups, rts, conditions):
		
		n_subjects = groups.shape[0]
		n_all_trials = rts.shape[1]
		n_conditions = len(set(conditions.flatten()))

		with open(f'{model_chain_name}.data', 'w') as data_of:	
			data_of.write('"nSubjects" <-\n')
			data_of.write(f'{n_subjects}\n')
			data_of.write('"nAllTrials" <-\n')
			data_of.write(f'{n_all_trials}\n')
			data_of.write('"nConditions" <-\n')
			data_of.write(f'{n_conditions}\n')
			data_of.write('"groupList" <-\n')
			data_of.write(self.__format_array_to_str(groups))
			data_of.write('"y" <-\n')
			data_of.write(self.__format_array_to_str(rts))
			data_of.write('"condList" <-\n')
			data_of.write(self.__format_array_to_str(conditions))

	def __format_array_to_str(self, arr) -> str:
		if len(arr.shape) == 1:
			res = 'c('
			res += ','.join([str(value) for value in arr])
			return res + ')\n'	
		else:
			dim = str(arr.shape)
			res = ','.join([str(value) if str(value) != 'nan' else 'NA' for value in arr.flatten(order='F')])
			return 'structure(c(' + res + f'),.Dim=c{dim})\n'				

	def _write_seed_data(self, model_chain_name):
		with open(f'{model_chain_name}.seed', 'w') as data_of:	
			data_of.write('".RNG.name" <- "base::Mersenne-Twister"\n".RNG.seed" <-\n')
			data_of.write(f'{int(random.random()*10000)}\n')

	def _write_init_data(self, model_chain_name, inits: dict):
		with open(f'{model_chain_name}.init', 'w') as data_of:	
			for param_init_name, param_init_arr in inits.items():
				data_of.write(f'"{param_init_name}" <-\n')
				data_of.write(self.__format_array_to_str(param_init_arr))

	def _write_model_file(self, model_chain_name, model_str, rts):

		model_str = model_str.replace('RT_MAX', str(np.nanmax(np.abs(rts.flatten()))+.01))

		with open(f'{model_chain_name}.jags', 'w') as data_of:	
			data_of.write(model_str)

	def _write_script(self, model_chain_name, nburn, niter, nthin, model_params):
		with open(f'{model_chain_name}.script', 'w') as data_of:
			data_of.write('load dic\n')
			data_of.write('load wiener\n')
			data_of.write(f'model in "{model_chain_name}.jags"\n')
			data_of.write(f'data in "{model_chain_name}.data"\n')
			data_of.write(f'compile, nchains(1)\n')
			data_of.write(f'parameters in "{model_chain_name}.init"\n')
			data_of.write(f'parameters in "{model_chain_name}.seed"\n')
			data_of.write(f'initialize\n')
			data_of.write(f'update {nburn}\n')
			for param in model_params:
				data_of.write(f'monitor set {param}, thin({nthin})\n')
			data_of.write(f'monitor deviance\n')
			niter_step = int(niter/20)
			for i in range(20):
				data_of.write(f'update {niter_step}\n')
			data_of.write(f"coda *, stem('{model_chain_name}_')\n")

	def combine_chains(self, model_name):
		model_dir = os.path.join(self._work_dir, model_name)
		os.chdir(model_dir)

		all_chains = glob.glob(f'{model_name}*_chain1.hdf5')
		logger.info(f"Combining chain files: {all_chains} ...")
		logger.info(all_chains)

		n_chains = len(all_chains)
		ofs = []
		for chain in all_chains:
			logger.debug(f'Reading chain {chain} ...')
			ofs.append(h5py.File(chain, 'r'))

		combined_chains = h5py.File(f'{model_name}.hdf5', 'w')

		try:	
			for key in ofs[0].keys():
				logger.debug(f'Reading {key} ... ')

				shape = list(ofs[0].get(key).shape)
				shape.insert(0, n_chains)
				
				combined_arr = np.full(shape, np.nan)

				for i, chain_of in enumerate(ofs):
					chain_idx = [i] + [slice(None)]*len(chain_of.get(key).shape)
					combined_arr[tuple(chain_idx)] = np.array(chain_of.get(key))

				combined_chains[key] = combined_arr

		except:
			logger.exception('Error!')
		finally:
			for of in ofs:
				logger.debug(f'Closing hd5 chain {of} ...')
				of.close()
			combined_chains.close()

		logger.info('Done combining!')

	def analyze_convergence(self, model_name, method='identity'):
		model_dir = os.path.join(self._work_dir, model_name)
		os.chdir(model_dir)

		model_chains = h5py.File(f'{model_name}.hdf5', 'r')

		if not os.path.isdir('convergence_plots'):
			os.makedirs('convergence_plots')
		os.chdir('convergence_plots')

		try:	
			for key in model_chains.keys():
				logger.debug(f'Reading {key} ... ')

				data = np.array(model_chains.get(key))

				for dim in self.__dim_iter(data.shape):				
					x = data[tuple(dim)]
					#x = np.squeeze(x)
					rhat = gelman_rubin(x, method=method)
					if rhat > 1.05:
						logger.info(f'Rhat > 1.05 found ({rhat:.3f}): {key}, index: {[d for d in dim if type(d) == int]}')
						fname = f'{model_name}__{key}_{"_".join([x.strip("[").strip("]") for x in str(dim).split(", ")])}.png'
						fname = fname.replace('_slice(None_None_None)', '')
						self.__plot_trace(x, fname)
		except:
			logger.exception('Error!')
		finally:
			model_chains.close()

	def __plot_trace(self, chains, fname):
		plt.figure()
		for c in range(chains.shape[0]):
			plt.plot(np.arange(chains.shape[1]), chains[c,:]) 
		plt.savefig(fname)
		plt.close()

	def __dim_iter(self, shape):
		shape = list(shape)
		shape.pop(0)
		shape.pop(-1)

		if len(shape) == 0:
			yield [slice(None), slice(None)]
		else:
			all_possibles = self.__dim_expand(shape)
			for dim in all_possibles:
				yield [slice(None)] + list(dim) + [slice(None)]

	def __dim_expand(self, shape):
		expansion = [list(range(x)) for x in shape]
		return product(*expansion)


if __name__ == '__main__':
	dm = DiffusionModel(work_dir="/home/fourbolt/Documents/diffusion_workdir")
	
	dm.initialize(
		model_name = 'youngold',
		groups_file="../../example_inputs/groups.csv",
		rts_file="../../example_inputs/rts.csv",
		conditions_file="../../example_inputs/conditions.csv",
		niter=100,
		nthin=1
	)
	

	dm.execute('youngold_model_00', chain='all', parallel=True)

	dm.format_chains('youngold_model_00', niter=100, nthin=1, chain='all')

	dm.combine_chains('youngold_model_00')

	dm.analyze_convergence('youngold_model_00')
