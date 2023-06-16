export CUDA_VISIBLE_DEVICES=5,6,7
# export MASTER_PORT=12346
# export MASTER_ADDR=
cd ..
python -m torch.distributed.launch --nproc_per_node=3 --master_port=12346 train_multi.py \
    --epochs 100\
    --batch-size 5 \
    --learning-rate 1e-6 \
    --scale 0.5 \
    --validation 0.0 \
    --in_classes 15 \
    --out_classes 4 \
    --dir_img data/radar_npy/factors/ \
    --dir_mask data/radar_npy/ob/ \
    --dir_checkpoint res/load_41_6/checkpoint/ \
    --save_checkpoint 1 \
    --save_interval 1 \
    --log_dir res/load_41_6/runs/ \
    --load /home/lfr/mntc/Pytorch-UNet/res/load_41_4/checkpoint/checkpoint_epoch100.pth \

    