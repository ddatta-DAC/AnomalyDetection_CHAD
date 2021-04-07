CUDA_VISIBLE_DEVICES=1,2,3,0 python3 executor.py --DATA_SET us_import1 --num_runs 5  
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 executor.py --DATA_SET us_import2 --num_runs 5  
CUDA_VISIBLE_DEVICES=2,1,3,0 python3 executor.py --DATA_SET us_import3 --num_runs 5  
CUDA_VISIBLE_DEVICES=3,2,1,0 python3 executor.py --DATA_SET us_import4 --num_runs 5  
CUDA_VISIBLE_DEVICES=2,1,3,0 python3 executor.py --DATA_SET us_import5 --num_runs 5  
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 executor.py --DATA_SET us_import6 --num_runs 5  
