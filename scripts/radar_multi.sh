export CUDA_VISIBLE_DEVICES=1,2
# export MASTER_PORT=12346
# export MASTER_ADDR=
cd ..
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12347 train_multi.py \
    --epochs 1\
    --batch-size 2 \
    --learning-rate 1e-5 \
    --scale 0.5 \
    --validation 1.0 \
    --in_classes 15 \
    --out_classes 4 \
    --dir_img data/radar_npy/factors/ \
    --dir_mask data/radar_npy/ob/ \
    --dir_checkpoint res/41_3/checkpoint/ \
    --save_checkpoint 1 \
    --save_interval 1 \
    --log_dir res/41_3/runs/ \
    