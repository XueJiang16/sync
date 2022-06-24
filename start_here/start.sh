# python3 sleep.py
cd /mapai/haowenguo/code/SPL/jx/
python3 -m torch.distributed.launch --nproc_per_node=$HOST_GPU_NUM occupy.py

# cd /mapai/haowenguo/code/SPL/jx/ssl
# bash start.sh
# bash run2.sh
# pip install scikit-learn
# pip install tqdm
# cd pytorch-gradual-warmup-lr/
# python3 setup.py install
# cd ../ssl
# python3 -m torch.distributed.launch --nproc_per_node=4 \
#   contrast.py -F1 time_warp -F2 time_warp -g 4 >logs/$TJ_TASK_ID.log
# cd ../start_here
# cd ..
# python3 -m torch.distributed.launch --nproc_per_node=$HOST_GPU_NUM occupy.py

# cd /mapai/haowenguo/code/SPL/jx/gradnorm_ood
# # chmod 777 *.sh
# bash pip.sh
# bash run.sh
# bash run3.sh

# cd /mapai/haowenguo/code/SPL/jx/
# cd mmclassification
# pip install -e .
# bash start.sh
# bash ./tools/dist_train.sh custom_config/inat/inat_config.py $HOST_GPU_NUM
# cd big_transfer
# bash run.sh

