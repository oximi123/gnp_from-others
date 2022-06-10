#!/bin/bash
# python -m torch.distributed.launch --nproc_per_node=2 main_tensoRF.py ../dataset/raw_room --workspace ../results/tensoRF/room
MP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_tensoRF.py ../dataset/fox --workspace ../results/tensoRF/fox -O --gui
