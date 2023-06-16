export CUDA_VISIBLE_DEVICES=5
cd ..
python train.py \
    --epochs 10\
    --batch-size 1 \
    --learning-rate 1e-5 \
    --scale 0.5 \
    --validation 90.0 \
    --in_classes 15 \
    --out_classes 4 \
    --dir_img data/radar_npy/factors/ \
    --dir_mask data/radar_npy/ob/ \
    --dir_checkpoint res/checkpoint/323/ \
    --save_checkpoint 1 \