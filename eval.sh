#CUDA_VISIBLE_DEVICES=2 python evaluation.py --lr 0.005 --batch 50

CUDA_VISIBLE_DEVICES=1 python evaluation.py --lr 0.001 --dr 0.98 --ds 100 --batch 10 --keep 0.8 --is_sensibility 1 #>> log_kuwo/lr0.002dr0.98ds100b10k0.8scale& 
