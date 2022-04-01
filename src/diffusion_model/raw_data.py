import logging
logger = logging.getLogger("raw_data")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)s | %(levelname)s | %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)

import os
import pandas as pd

class RawData:
	def __init__(self, groups_file=None, rts_file=None, conds_file=None):

		if not os.path.isfile(groups_file):
			raise Exception(f'Groups csv does not exist! (Is the path: {groups_file} correct?)')
		if not os.path.isfile(rts_file):
			raise Exception(f'RTs csv does not exist! (Is the path: {rts_file} correct?)')
		if not os.path.isfile(conds_file):
			raise Exception(f'Conditions csv does not exist! (Is the path: {conds_file} correct?)')

		# Process groups		
		logger.debug("Reading groups data ...")
		groups = pd.read_csv(groups_file, index_col=0, float_precision='round_trip')
		if "group" not in groups.columns:
			raise Exception(f'{groups_file} does not have a "group" column!')
		groups = groups['group'].values
		if len(set(groups)) != 2:
			raise Exception(f"Invalid number of groups. Found: {len(set(groups))}, Expected: 2")
		self.groups = groups
		logger.debug(f"Found {len(self.groups)} subjects!")

		# Process rts
		logger.debug("Reading rts data ...")
		rts = pd.read_csv(rts_file, index_col=0, float_precision='round_trip')
		self.rts = rts.values
		logger.debug(f"Found {rts.shape[0]} subjects, and {self.rts.shape[1]} trials!")
			
		# Process conditions
		logger.debug("Reading conditions data ...")
		conditions = pd.read_csv(conds_file, index_col=0, float_precision='round_trip')
		self.conds = conditions.values
		logger.debug(f"Found {len(set(list(self.conds.flatten())))} conditions, {self.conds.shape[0]} subjects, and {self.conds.shape[1]} trials!")

		logger.info(f'Raw data processed: {self}')

	def __str__(self):
		return f'RawData(groups={self.groups.shape},rts={self.rts.shape},conds={self.conds.shape})'