#CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.001 --dr 0.98 --ds 100 --batch 10 --keep 0.8 >> log_kuwo/lr0.001dr0.98ds100b50k0.8& 
#CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.002 --dr 0.98 --ds 100 --batch 10 --keep 0.8 >> log_kuwo/lr0.002dr0.98ds100b10k0.8& 
#CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.003 --dr 0.98 --ds 100 --batch 50 --keep 0.8 >> log_kuwo/lr0.003dr0.98ds100b50k0.8& 
#CUDA_VISIBLE_DEVICES=2 python train.py --lr 0.004 --dr 0.98 --ds 100 --batch 50 --keep 0.8 >> log_kuwo/lr0.004dr0.98ds100b50k0.8& 

CUDA_VISIBLE_DEVICES=1 python train.py --lr 0.001 --dr 0.98 --ds 100 --batch 10 --keep 0.8 --is_sensibility 1 >> log_kuwo/sensibility/lr0.001dr0.98ds100b10k0.8scale& 




