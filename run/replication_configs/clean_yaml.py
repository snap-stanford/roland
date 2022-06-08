"""
Collect optimal model config files from each hyperparameter tuning session.
"""
import os
import sys
import yaml
from tqdm import tqdm


DATASET_DIR = '/lfs/hyperturing2/0/tianyudu/all_datasets'
OUT_DIR = 'results'


if __name__ == "__main__":
	config_path, out_path, REMARK = sys.argv[1], sys.argv[2], sys.argv[3]
	config_files = [x for x in os.listdir(config_path) if x.endswith('.yaml')]
	for config_file in tqdm(config_files):
		with open(os.path.join(config_path, config_file)) as f:
			config = yaml.safe_load(f)

		del config['cfg_dest']
		del config['bn']
		del config['dataset']['cache_load']
		del config['dataset']['cache_save']
		del config['dataset']['edge_message_ratio']
		del config['dataset']['edge_negative_sampling_ratio']
		del config['dataset']['num_nodes']

		del config['params']
		del config['print']
		del config['round']

		# Modify yaml here.
		config['dataset']['edge_dim'] = 0
		config['dataset']['format'] = 'infer'
		config['remark'] = REMARK
		config['out_dir'] = OUT_DIR
		config['device'] = 'auto'
		config['dataset']['dir'] = DATASET_DIR
		# Save modified yaml.
		with open(os.path.join(out_path, config_file), 'w') as f:
			yaml.dump(config, f)
