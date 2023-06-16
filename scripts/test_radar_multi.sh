export CUDA_VISIBLE_DEVICES=3,4
# export MASTER_PORT=12346
# export MASTER_ADDR=
cd ..
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 test_multi.py \
    --batch-size 2 \
    --in_classes 15 \
    --out_classes 4 \
    --dir_img data/radar_npy/factors/ \
    --dir_mask data/radar_npy/ob/ \
    --dir_checkpoint res/metrice_load414_100_2/checkpoint/ \
    --log_dir res/metrice_load414_100_2/runs/ \
    --load /home/lfr/mntc/SEResNet/res/41_3/checkpoint/checkpoint_epoch1.pth \

    