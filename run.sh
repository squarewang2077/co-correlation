#!/bin/bash


# for w in 16 32 64 128 256 512 1024 2048 4096 8192
# do 
#     for act in 'linear' 'relu'
#     do
#         echo $w $act
#         python bianalyze_shallow_model_on_small_dataset.py --width $w --act_name $act  
#     done
# done
 
python bianalyze_resnet50_on_cifar10.py 
