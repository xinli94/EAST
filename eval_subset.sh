ROOT_PATH='/data5/xin/ocr'
DATASET='rot_boxes_v2_normal_with_class'
DATA_CSV='val.csv'
GPU_ID=6

# BACKBONE='resnet_v1_50'
# CKPT_PATH='pretrained'
# ID='east_icdar2015_resnet_v1_50_rbox'

BACKBONE='resnet_v1_50'
CKPT_PATH='train'
ID='res50_0228'

# BACKBONE='resnet_v1_101'
# CKPT_PATH='train'
# ID='res101_0228'

# run eval
echo 'Step[1/3]: run eval'
python eval.py \
--test_data_path=$ROOT_PATH/$DATASET/$DATA_CSV \
--gpu_list=$GPU_ID \
--checkpoint_path=$ROOT_PATH/$CKPT_PATH/$ID \
--output_dir=$ROOT_PATH/output/val_$ID/ \
--score_threshold 0.001 \
--vis_threshold 0.001 \
--backbone $BACKBONE

# convert to csvn
echo 'Step[2/3]: convert'
python convert.py \
--type csvn \
--input_folder $ROOT_PATH/output/val_$ID/ \
--output_file $ROOT_PATH/data_csv/val_result_$ID.csv

# precision recall
echo 'Step[3/3]: run metrics'
cd scripts && bash pr.sh $ROOT_PATH/data_csv/val_result_$ID.csv $ROOT_PATH/data_csv/groundtruth_val.csv val_$ID