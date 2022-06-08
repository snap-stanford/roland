#!/usr/bin/env bash
# This script replicates tables reported in the roland paper.
# Author: Tianyu Du
# Last modified: June 8, 2022

# put the path of dataset here.
DATA_PATH='/home/tianyudu/Data/roland_public_data'
# DATA_PATH='/lfs/hyperturing1/0/tianyudu/roland_public_data'

# =============================================================================
# Table 2.
# =============================================================================
for file in $(ls -d ./replication_configs/table2/*.yaml)
    do
    python3 main.py --cfg ${file} --repeat 3 --override_data_dir ${DATA_PATH} --override_remark 'table2'
done

# =============================================================================
# Table 3 Top Panel.
# =============================================================================
for file in $(ls -d ./replication_configs/table3_top/*.yaml)
    do
    python3 main.py --cfg ${file} --repeat 1 --override_data_dir ${DATA_PATH} --override_remark 'table3_top'
done

# =============================================================================
# Table 3 Middle Panel.
# =============================================================================
for file in $(ls -d ./replication_configs/table3_middle/*.yaml)
    do
    python3 main.py --cfg ${file} --repeat 1 --override_data_dir ${DATA_PATH} --override_remark 'table3_middle'
done

# =============================================================================
# Table 3 Bottom Panel.
# =============================================================================
for file in $(ls -d ./replication_configs/table3_bottom/*.yaml)
    do
    python3 main.py --cfg ${file} --repeat 1 --override_data_dir ${DATA_PATH} --override_remark 'table3_bottom'
done

# =============================================================================
# Table 4.
# =============================================================================
for file in $(ls -d ./replication_configs/table4/*.yaml)
    do
    python3 main.py --cfg ${file} --repeat 1 --override_data_dir ${DATA_PATH} --override_remark 'table4'
done
