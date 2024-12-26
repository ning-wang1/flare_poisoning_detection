#!/bin/sh
cd ..
cd ..

export CUDA_VISIBLE_DEVICES=0

# # attack krum

# python main.py --json_file script/bash_cifar/cifar.json --seed 1 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method fltrust 
# python main.py --json_file script/bash_cifar/cifar.json --seed 1 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method diamond
# python main.py --json_file script/bash_cifar/cifar.json --seed 1 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method contra
# python main.py --json_file script/bash_cifar/cifar.json --seed 1 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method brar --gar krum
# python main.py --json_file script/bash_cifar/cifar.json --seed 1 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method brar --gar coomed
# python main.py --json_file script/bash_cifar/cifar.json --seed 1 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect_method none --gar avg
# python main.py --json_file script/bash_cifar/cifar.json --seed 1 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method brar --gar trimmedmean

# python main.py --json_file script/bash_cifar/cifar.json --seed 2 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method fltrust 
python main.py --json_file script/bash_cifar/cifar.json --seed 2 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method diamond
python main.py --json_file script/bash_cifar/cifar.json --seed 2 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method contra
# python main.py --json_file script/bash_cifar/cifar.json --seed 2 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method brar --gar krum
# python main.py --json_file script/bash_cifar/cifar.json --seed 2 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method brar --gar coomed
# python main.py --json_file script/bash_cifar/cifar.json --seed 2 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect_method none --gar avg
# python main.py --json_file script/bash_cifar/cifar.json --seed 2 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method brar --gar trimmedmean

# python main.py --json_file script/bash_cifar/cifar.json --seed 3 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method fltrust 
# python main.py --json_file script/bash_cifar/cifar.json --seed 3 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method diamond
# python main.py --json_file script/bash_cifar/cifar.json --seed 3 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method contra
# python main.py --json_file script/bash_cifar/cifar.json --seed 3 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method brar --gar krum
# python main.py --json_file script/bash_cifar/cifar.json --seed 3 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method brar --gar coomed
# python main.py --json_file script/bash_cifar/cifar.json --seed 3 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect_method none --gar avg
# python main.py --json_file script/bash_cifar/cifar.json --seed 3 --lr_reduce --train --mal --attack_type backdoor_krum_duplicate --mal_strat converge_train_alternate_dist_self --detect --detect_method brar --gar trimmedmean

