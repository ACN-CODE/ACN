CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 main.py --folder ./experiments/capsulexlan
