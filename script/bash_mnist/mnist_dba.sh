#!/bin/sh
cd ..
cd ..

export CUDA_VISIBLE_DEVICES=3

# # attack krum

python main.py --json_file script/bash_mnist/mnist.json --seed 1 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method fltrust 
python main.py --json_file script/bash_mnist/mnist.json --seed 1 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method diamond
python main.py --json_file script/bash_mnist/mnist.json --seed 1 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method contra
python main.py --json_file script/bash_mnist/mnist.json --seed 1 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method brar --gar krum
python main.py --json_file script/bash_mnist/mnist.json --seed 1 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method brar --gar coomed
python main.py --json_file script/bash_mnist/mnist.json --seed 1 --lr_reduce --train --mal --attack_type dba_backdoor --detect_method none --gar avg
python main.py --json_file script/bash_mnist/mnist.json --seed 1 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method brar --gar trimmedmean

python main.py --json_file script/bash_mnist/mnist.json --seed 2 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method fltrust 
python main.py --json_file script/bash_mnist/mnist.json --seed 2 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method diamond
python main.py --json_file script/bash_mnist/mnist.json --seed 2 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method contra
python main.py --json_file script/bash_mnist/mnist.json --seed 2 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method brar --gar krum
python main.py --json_file script/bash_mnist/mnist.json --seed 2 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method brar --gar coomed
python main.py --json_file script/bash_mnist/mnist.json --seed 2 --lr_reduce --train --mal --attack_type dba_backdoor --detect_method none --gar avg
python main.py --json_file script/bash_mnist/mnist.json --seed 2 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method brar --gar trimmedmean

python main.py --json_file script/bash_mnist/mnist.json --seed 3 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method fltrust 
python main.py --json_file script/bash_mnist/mnist.json --seed 3 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method diamond
python main.py --json_file script/bash_mnist/mnist.json --seed 3 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method contra
python main.py --json_file script/bash_mnist/mnist.json --seed 3 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method brar --gar krum
python main.py --json_file script/bash_mnist/mnist.json --seed 3 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method brar --gar coomed
python main.py --json_file script/bash_mnist/mnist.json --seed 3 --lr_reduce --train --mal --attack_type dba_backdoor --detect_method none --gar avg
python main.py --json_file script/bash_mnist/mnist.json --seed 3 --lr_reduce --train --mal --attack_type dba_backdoor --detect --detect_method brar --gar trimmedmean


