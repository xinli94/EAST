OUT_FILE=$1
PRED_FILE=$2
GT_FILE=$3

python pr_curve/pr_curve_rbox.py $PRED_FILE $GT_FILE | sort -k1,1 -t, -g -r > $OUT_FILE
TOTAL=`cat $GT_FILE | wc -l`
echo $GT_FILE': '$TOTAL
python pr_curve/roc.py $OUT_FILE $TOTAL > $OUT_FILE.roc
