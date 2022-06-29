#!/bin/sh

# eval "$(conda shell.bash hook)"
PYTHON=python
TEST_CODE=test_ipadscan.py

exp_dir=$1
model_dir=${exp_dir}/model
result_dir=${model_dir}/result
config=config/ipad_scaned/config.yaml

mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

cp ${config} tool/test_ipadscan.sh tool/${TEST_CODE} ${exp_dir}
cp tool/${TEST_CODE} ${exp_dir}/code/tool/

export PYTHONPATH=${exp_dir}/code
now=$(date +"%Y%m%d_%H%M%S")

#: '
$PYTHON -u ${exp_dir}/code/tool/${TEST_CODE} \
  --config=${config} \
  save_folder ${result_dir}/last \
  model_path ${model_dir}/model_last.pth \
  2>&1 | tee ${exp_dir}/test_$now-last-ipadscan.log
#'
