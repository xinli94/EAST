DEFAULT='/data5/xin/object_dectection_hookit57/fixed.csv'
PRED_FILE=${1:-$DEFAULT}

DEFAULT='/data5/xin/irv2_atrous/test.txt'
GT_FILE=${2:-$DEFAULT}

DEFAULT='out'
OUT_FILE=${3:-$DEFAULT}

DEFAULT='pr'
OUT_IMAGE=${4:-$DEFAULT}

bash pr_curve/pr_curve.sh $OUT_FILE $PRED_FILE $GT_FILE
# rm -f roc_results/*
# mkdir -p roc_results/
mv $OUT_FILE* roc_results/

rm -f $OUT_IMAGE.png
python pr_curve/plot.py roc_results $OUT_IMAGE
echo 'pr curve image: '$OUT_IMAGE.png
