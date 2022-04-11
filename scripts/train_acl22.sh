#!/bin/bash

codebase=`pwd`
loggingdir=./logs
mkdir -p $loggingdir

cd $loggingdir
lang_config=$1
phi=$2 #phi or non-acl training method
transformer=$3 #xlm-roberta or mbert

if [[ $phi == 'uniform' ]]
    then param_config=$codebase/configs/params.mbert.uniform.json;
elif [[ $phi == 'sizeprop' ]]
    then param_config=$codebase/configs/params.mbert.json;
elif [[ $phi == 'smoothSampling' ]]
    then param_config=$codebase/configs/params.mbert.smoothSampling.json;
else
    param_config=$codebase/configs/params.mbert.acl.json 
fi

SEED=$RANDOM \
BATCH_SIZE=32 \
MAX_SENTS=15000 \
PHI=$phi \
EPOCHS=80 \
python $codebase/train.py --parameters_config $param_config --dataset_config $codebase/configs/$lang_config.json --device 0 --name $lang_config\_$transformer\_$phi
