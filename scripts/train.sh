#!/bin/bash
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
python ./main.py train --model_name minist_classifier --criterion cross_entropy --dataset minist_dataset --lr 1e-3 --epochs 10 --batch_size 128 --seed 42 --project_root_path /data/rms/test/ --print_by_step 8 --log_to_file
# python ./main.py train --model_name minist_classifier --criterion cross_entropy --dataset minist_dataset --lr 1e-3 --epochs 100 --batch_size 64 --seed 42 --project_root_path ./ --print_by_step 8 --log_to_file

