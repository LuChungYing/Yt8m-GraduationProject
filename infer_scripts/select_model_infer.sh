model=$1
echo $model
python inference.py \
    --train_dir="/home/lulu/yt8m/models/frame/${model}" \
    --output_file='/home/lulu/yt8m/output/dbof.csv'\
    --input_data_pattern=/home/lulu/yt8m/3/frame/test/test*.tfrecord \
    --batch_size=32 \
    --segment_labels