CUDA_VISIBLE_DEVICES=1,2,3,0 python3 main.py --DIR us_import1 --num_runs 5  --objective soft-boundary &
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 main.py --DIR us_import2 --num_runs 5  --objective soft-boundary & 
CUDA_VISIBLE_DEVICES=2,2,3,0 python3 main.py --DIR us_import3 --num_runs 5  --objective soft-boundary 
CUDA_VISIBLE_DEVICES=2,1,3,0 python3 main.py --DIR us_import4 --num_runs 5  --objective soft-boundary & 
CUDA_VISIBLE_DEVICES=3,2,1,0 python3 main.py --DIR us_import5 --num_runs 5  --objective soft-boundary  
CUDA_VISIBLE_DEVICES=3,2,1,0 python3 main.py --DIR us_import6 --num_runs 5  --objective soft-boundary & 

CUDA_VISIBLE_DEVICES=1,2,3,0 python3 main.py --DIR us_import1 --num_runs 5  --objective 'one-class' & 
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 main.py --DIR us_import2 --num_runs 5  --objective 'one-class' 
CUDA_VISIBLE_DEVICES=2,2,3,0 python3 main.py --DIR us_import3 --num_runs 5  --objective 'one-class' & 
CUDA_VISIBLE_DEVICES=2,1,3,0 python3 main.py --DIR us_import4 --num_runs 5  --objective 'one-class'  
CUDA_VISIBLE_DEVICES=3,2,1,0 python3 main.py --DIR us_import5 --num_runs 5  --objective 'one-class' & 
CUDA_VISIBLE_DEVICES=3,2,1,0 python3 main.py --DIR us_import6 --num_runs 5  --objective 'one-class'  
    
