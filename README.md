# Searching towards Class-Aware Generators for Conditional Generative Adversarial Networks [paper](https://arxiv.org/abs/2006.14208) 

## Updates

- [x] Evaluation code
- [ ] Search code (coming!)

## Environment 

* CUDA 10.0; cudnn-10.0-v7.6.5; Python == 3.6
* PyTorch == 1.4.0, torchvision == 0.5.0
```
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```
* detectron2
```
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html
```
* TensorFlow == 1.13.1
```
pip install tensorflow-gpu==1.13.1
```
* Additional used packages
```
pip install -r requirements.txt
```

## Clone git repository
```bash
git clone --recursive https://github.com/PeterouZh/NAS_cGAN.git
cd NAS_cGAN
```

## Prepare FID statistic file

Pre-calculate FID statistic files of CIFAR10 using PyTorch and TensorFlow models, respectively. 
Those files will be saved to *./datasets/nas_cgan/pytorch_fid_stat/fid_stats_pytorch_cifar10_train_32.npz* and 
*./datasets/nas_cgan/tf_fid_stat/fid_stats_tf_cifar10_train_32.npz*.

Or you can download our pre-calculated statistic files [onedrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/zhoupengcv_sjtu_edu_cn/EuR9yg6MUzpGr3H7RGQdEUQBnXz5qMpfL-tKNTvg-t4_XA?e=CGFZ42),
which should be put into the directories mentioned above.
 
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:./exp
python 	exp/nas_cgan/scripts/train_net.py \
  --config exp/nas_cgan/configs/calculate_fid_stat_CIFAR10.yaml \
  --command calculate_fid_stat_CIFAR10 \
  --outdir results/calculate_fid_stat_CIFAR10
```

## Evaluate the models reported in the paper

Download the pre-trained models [onedrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/zhoupengcv_sjtu_edu_cn/EsokPqpwPMhPi8IjPh8WQBoBQF9S1iunCj-EdpawvjyyHQ?e=owFvIe), 
and put them into *./datasets/nas_cgan/models*.
### FID and IS
Eval NAS-cGAN on CIFAR10.
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:./exp
python 	exp/nas_cgan/scripts/train_net.py \
  --config exp/nas_cgan/configs/eval_NAS_cGAN_CIFAR10.yaml \
  --command eval_NAS_cGAN_CIFAR10 \
  --outdir results/eval_NAS_cGAN_CIFAR10
```

Eval NAS-caGAN on CIFAR10.
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:./exp
python 	exp/nas_cgan/scripts/train_net.py \
  --config exp/nas_cgan/configs/eval_NAS_caGAN_CIFAR10.yaml \
  --command eval_NAS_caGAN_CIFAR10 \
  --outdir results/eval_NAS_caGAN_CIFAR10
```

### intra FIDs

Pre-calculate intra FID statistic of each class on CIFAR10. 
Or you can use our pre-calculated files [onedrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/zhoupengcv_sjtu_edu_cn/EhWbm-z9lLJDpcZ5KuqmfO0Bd5ak80J5QBT_G3y6zkYdEw?e=PR3VEF),
and put these files in *./datasets/nas_cgan/tf_fid_stat/cifar10_train_per_class_32*.
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:./exp
python 	exp/nas_cgan/scripts/train_net.py \
  --config exp/nas_cgan/configs/calculate_fid_stat_per_class_CIFAR10.yaml \
  --command calculate_fid_stat_per_class_CIFAR10 \
  --outdir results/calculate_fid_stat_per_class_CIFAR10
``` 

Eval intra FIDs for NAS-cGAN.
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:./exp
python 	exp/nas_cgan/scripts/train_net.py \
  --config exp/nas_cgan/configs/eval_intra_FID_NAS_cGAN_CIFAR10.yaml \
  --command eval_intra_FID_NAS_cGAN_CIFAR10 \
  --outdir results/eval_intra_FID_NAS_cGAN_CIFAR10
```

Eval intra FIDs for NAS-caGAN.
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:./exp
python 	exp/nas_cgan/scripts/train_net.py \
  --config exp/nas_cgan/configs/eval_intra_FID_NAS_caGAN_CIFAR10.yaml \
  --command eval_intra_FID_NAS_caGAN_CIFAR10 \
  --outdir results/eval_intra_FID_NAS_caGAN_CIFAR10
```

### The calibrated results

Prepare intra FID statistic files following the steps above (put these files in *./datasets/nas_cgan/tf_fid_stat/cifar10_train_per_class_32*).
Download FID logits files  [onedrive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/zhoupengcv_sjtu_edu_cn/ElDGDTDcfe9JnYV4sfsFRPwBpgA4UZZDrHKv80F3aZy3JQ?e=YIFdgP) produced by the calibrated models,
and put the files into *./datasets/nas_cgan/fid_logits_of_calibrated_model/*.

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:./exp
python 	exp/nas_cgan/scripts/train_net.py \
  --config exp/nas_cgan/configs/eval_calibrated_NAS_cGAN.yaml \
  --command eval_calibrated_NAS_cGAN \
  --outdir results/eval_calibrated_NAS_cGAN
```

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/cudnn-10.0-v7.6.5.32/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:./exp
python 	exp/nas_cgan/scripts/train_net.py \
  --config exp/nas_cgan/configs/eval_calibrated_NAS_caGAN.yaml \
  --command eval_calibrated_NAS_caGAN \
  --outdir results/eval_calibrated_NAS_caGAN
```

## Acknowledgement

1. https://github.com/facebookresearch/detectron2
1. https://github.com/TDeVries/enas_pytorch
1. https://github.com/TAMU-VITA/AutoGAN
1. https://github.com/ajbrock/BigGAN-PyTorch
1. https://github.com/rosinality/stylegan2-pytorch
1. https://github.com/openai/improved-gan/tree/master/inception_score
1. https://github.com/bioinf-jku/TTUR
