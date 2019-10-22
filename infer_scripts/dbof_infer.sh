python inference.py \
    --train_dir='models/frame/dbof' \
    --output_file='output/dbof.csv'\
    --input_data_pattern=/home/lulu/yt8m/3/frame/test/test*.tfrecord \
    --segment_labels \
    --batch_size 32 \