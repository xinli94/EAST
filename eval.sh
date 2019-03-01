ROOT_PATH='/data5/xin/ocr'
DATASET='rot_boxes_v2_normal_with_class'

# CKPT='east_icdar2015_resnet_v1_50_rbox'
# GPU_ID=0
# BACKBONE='resnet_v1_50'

# CKPT='east_icpr2018_resnet_v1_50_rbox_1035k'
# GPU_ID=8
# BACKBONE='resnet_v1_50'

CKPT='east_mix2018_resnet_v1_101_rbox_987k'
GPU_ID=9
BACKBONE='resnet_v1_101'

# ID=''
ID='_box_threshold_0'

# run eval
echo 'Step[1/3]: run eval'
python eval.py \
--test_data_path=$ROOT_PATH/$DATASET/images/ \
--gpu_list=$GPU_ID \
--checkpoint_path=$ROOT_PATH/pretrained/$CKPT/ \
--output_dir=$ROOT_PATH/output/$CKPT$ID/ \
--score_threshold 0.001 \
--backbone $BACKBONE

# convert to csvn
echo 'Step[2/3]: convert'
python convert.py \
--type csvn \
--input_folder $ROOT_PATH/output/$CKPT$ID/ \
--output_file $ROOT_PATH/data_csv/result_$CKPT$ID.csv

# precision recall
echo 'Step[3/3]: run metrics'
cd scripts && bash pr.sh $ROOT_PATH/data_csv/result_$CKPT$ID.csv $ROOT_PATH/data_csv/groundtruth.csv $CKPT$ID
