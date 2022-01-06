dataset="potsdam"

CUDA_VISIBLE_DEVICES='0' python -u Eval.py \
    --backbone 'resnet18' \
    --model "BESNet" \
    --num_classes 6 \
    --batch_size 1 \
    --num_workers 1 \
    --overlap 0.33333 \
    --flip True \
    --scales 0.75 1.0 1.25 \
    --save_flag False \
    --save_dir "./preds/${dataset}/2022.01.01/" \
    --dataset $dataset \
    --data_dir "/media/FlyC235/Fly/dataset/${dataset}/" \
    --test_list  "/media/FlyC235/Fly/dataset/${dataset}/${dataset}_test_fullsize.txt" \
    --checkpoint "/media/FlyC235/Fly/BESnet/snapshots/potsdam/BESNet_0.005_4/2022-01-01_11-11-11/checkpoint/best_metrics.pth" \