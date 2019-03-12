#!/bin/bash
ds=$1
col=$2
dim=$3
epoch=$4
mode=$5
spec=
#spec=-full
words=2
v=$ds
if [[ $pt ]]; then
    pt="-pretrainedVectors $pt"
fi
model=${col}${spec}.${dim}d.${epoch}e
trainFile=$v/train/${col}.full
if [[ -e $v/train/$col.small ]]; then 
    trainExt=small
else
    trainExt=full
fi
if [[ -e $v/test/$col.small ]]; then 
    testExt=small
else
    testExt=full
fi
str="$col with $dim dimensions for $epoch epochs"
if [[ $mode == 'new' && -e $v/models/$model.bin ]]; then echo "Skip training $str, file exists"; exit; fi
if [[ -s $v/models/$model.bin ]]; then echo "Skip training $str, already trained"; exit; fi
if [[ ! -e $v/train/$col.$trainExt || ! -e $trainFile || ! -e $v/test/$col.$testExt ]]; then echo "Skip training $str, missing files"; exit; fi
echo "Training $str..."
../fasttext supervised $pt -input $trainFile -output $v/models/$model -epoch $epoch -dim $dim -wordNgrams $words -lr 1 2>&1 | tee $v/results/$model.log
echo "Running tests for $str..."
#(../fasttext test $v/models/$model.bin $v/test/$col.$perc | tee $v/results/$model.test.$perc &)
#(../fasttext test $v/models/$model.bin $v/train/$col.$perc | tee $v/results/$model.train.$perc &)
../fasttext test $v/models/$model.bin $v/test/$col.$testExt | tee $v/results/$model.test
../fasttext test $v/models/$model.bin $v/train/$col.$trainExt | tee $v/results/$model.train
