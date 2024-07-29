#!/bin/bash

# Script for the experiment shown in Figure 10 of the main body of the paper.
python mainCNF.py --dataset mnist_vae --mean_r 10 --file_name example1 --epochs 5

# Script for the experiment shown in Figure 11 of the main body of the paper (our proposed method).
python mainCNF.py --dataset mnist --mean_r 10 --file_name example2 --epochs 5

# Script for the experiment shown in Figure 11 of the main body of the paper (baseline methods).
python mainGDA.py --dataset portraits --method sourceonly --file_name example3 --epochs 5

# Script for the experiment shown in Figure 7 of the main body of the parper.
python mainCNF.py --dataset block --mean_r 10 --no_inter True --label_ratio 100-0-0-0 --file_name example4 --epochs 5
python mainCNF.py --dataset block --mean_r 10 --no_inter False --label_ratio 100-0-0-0 --file_name example5 --epochs 5

# Script for the experiment shown in Figure 3 of the main body of the paper.
python mainCNF.py --dataset moon --mean_r 3 --file_name example6 --epochs 3
python mainCNF.py --dataset moon --mean_r 3 --log_prob_method gmm --log_prob_param 30 30 --file_name example7 --epochs 5

# Script for the experiment shown in 5.5. Hyperparameters.
# Hyperparameter k
python mainCNF.py --dataset rxrx1 --mean_r 10 --log_prob_param 5 5 --file_name example8 --epochs 5
# Hyperparameter r
python CrossValidation.py --dataset rxrx1 --mean_r 1 --source_only True --file_name example9 --epochs 5
