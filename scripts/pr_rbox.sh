DEFAULT='/data5/xin/ocr/ms_ocr_boxes_with_hive_cat/east_output/output_rbox.txt'
PRED_FILE=${1:-$DEFAULT}

DEFAULT='/data5/xin/ocr/ms_ocr_boxes_with_hive_cat/gt_val.csv'
GT_FILE=${2:-$DEFAULT}

DEFAULT='out'
OUT_FILE=${3:-$DEFAULT}

DEFAULT='pr'
OUT_IMAGE=${4:-$DEFAULT}

DEFAULT='roc_results_r'
ROC=${5:-$DEFAULT}

bash pr_curve/pr_curve_rbox.sh $OUT_FILE $PRED_FILE $GT_FILE
# rm -f roc_results/*
# mkdir -p roc_results/
mv $OUT_FILE* $ROC

# rm -f $OUT_IMAGE.png
python pr_curve/plot.py $ROC $OUT_IMAGE
echo 'pr curve image: '$OUT_IMAGE.png
