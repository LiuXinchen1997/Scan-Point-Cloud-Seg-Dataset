#!/bin/sh

# eval "$(conda shell.bash hook)"
PYTHON=python

dataset=$1
exp_name=$2

TRAIN_CODE=train.py
TEST_CODE=test.py

exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

code_dir=${exp_dir}/code
mkdir -p ${code_dir}
cp -r tool/ util/ model/ config/ lib/ ${code_dir}
cp ${config} tool/train.sh tool/${TRAIN_CODE} ${exp_dir}

export PYTHONPATH=${code_dir}
now=$(date +"%Y%m%d_%H%M%S")
#$PYTHON ${exp_dir}/${TRAIN_CODE} \
#  --config=${config} \
#  save_path ${exp_dir} \
#  2>&1 | tee ${exp_dir}/train-$now.log

$PYTHON ${exp_dir}/${TRAIN_CODE} --config=${config} save_path ${exp_dir} 2>&1 | tee ${exp_dir}/train-$now.log
