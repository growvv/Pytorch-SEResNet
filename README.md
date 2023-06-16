## Quick Start
1. [安装 CUDA](https://developer.nvidia.com/cuda-downloads)

2. [安装 PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. 安装依赖

`pip install -r requirements.txt`

4. 进行训练

- 单GPU版本

`bash scripts/radar.sh`

- 多GPU版本

`bash scripts/radar_multi.sh`

5. 进行测试

`bash scripts/test_radar_multi.sh`

## 数据集
- 输入格式：in_classes x H x W

- 输出格式：out_classes x H x W

例如，在降水定量估计任务中，
- 输入格式：npy文件，shape为(15, 256, 256)，15表示15个因子，256表示图像的高度，256表示图像的宽度。

- 输出格式：npy文件，shape为(4, 256, 256)，4表示4个降水等级，256表示图像的高度，256表示图像的宽度。

## 代码结构
- `train.py`：训练模型，单GPU版本。
- `train_multi.py`：训练模型，多GPU版本。
- `evaluate.py`：验证模型，在训练神经网络模型的过程中对模型在验证集上的性能进行评估
- `utils`：工具包，包含数据预处理、数据增强、模型评估等功能。
- `unet`：模型的网络结构
- `res`：保存训练过程中的模型权重、日志文件等。
- `scripts`：训练和测试的脚本。
- `data`：数据集。

## 训练和预测

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12347 train_multi.py \
    --epochs 3\
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
```

- --master_port=12347：设置主节点的通信端口。

- --epochs 3：训练过程要遍历整个训练数据集的次数。

- --batch-size 2：在训练过程中，一次输入到网络中并更新参数的样本数量。

- --learning-rate 1e-5：学习率，决定参数更新的快慢。

- --scale 0.5：这可能是数据预处理的一个步骤，可能是缩放图像的因子。

- --validation 1.0：验证集的比例或使用验证的频率。

- --in_classes 15：输入的类别数。

- --out_classes 4：输出的类别数。

- --dir_img data/radar_npy/factors/：输入的多因子数据的文件夹路径。

- --dir_mask data/radar_npy/ob/：输入的目标数据的文件夹路径。

- --dir_checkpoint res/41_3/checkpoint/：在训练过程中，模型权重保存的文件夹路径。

- --save_checkpoint 1：是否保存模型权重的标识，1表示保存，0可能表示不保存。

- --save_interval 1：保存模型权重的频率，例如，每过1个epoch保存一次模型权重。

- --log_dir res/41_3/runs/：日志文件保存的文件夹路径。

