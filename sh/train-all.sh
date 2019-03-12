# 4 epoch > 3 epoch > 2 epoch > 1 epoch
# 32d > 16d > 8d
# 64d ~ 32d

ds=$1
cols=$(./list.sh ${ds})
for col in $cols; do
    #./once.sh ./train.sh $col 16 4
    #./once.sh ./train.sh $ds $col 16 6
    #./once.sh ./train.sh $ds $col 20 6
    ./once.sh ./train.sh $ds $col 64 20
    #./once.sh ./train.sh $col 32 4
done

#./once.sh ./train.sh TradeName 32 8
#./once.sh ./train.sh TradeName 64 4
#./once.sh ./train.sh TradeName 16 4
#./once.sh ./train.sh TradeName 64 8
#./once.sh ./train.sh TradeName 96 4
#./once.sh ./train.sh TradeName 96 8
