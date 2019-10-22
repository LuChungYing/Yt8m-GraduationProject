model=$1
echo $model
python eval.py \
    --train_dir="/home/lulu/yt8m/models/frame/${model}" \
    --eval_data_pattern='/home/lulu/yt8m/3/frame/validate/validate*.tfrecord' \
    --frame_features \
    --batch_size=32 \
    --segment_lables