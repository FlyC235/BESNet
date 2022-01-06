set -e
Model="BESNet"
dataset="potsdam"

CUDA_VISIBLE_DEVICES='0' python -u train_fcn.py \
    --model $Model \
    --backbone 'resnet18' \
    --aux_classifier True \
    --auxloss_weight 0.4 \
    --num_classes 6 \
    --batch_size 4 \
    --dataset $dataset \
    --save_dir './snapshots/' \
    --data_dir "/media/FlyC235/Fly/dataset/${dataset}/" \
    --train_list "/media/FlyC235/Fly/dataset/${dataset}/train.txt" \
    --test_list  "/media/FlyC235/Fly/dataset/${dataset}/test.txt" \
    --learning_rate 5e-3 \
    --weight_decay 5e-4 \
    --num_workers 4 \
    --max_epoches 200 \
    --warmup_epochs 0 \
    --no-val False