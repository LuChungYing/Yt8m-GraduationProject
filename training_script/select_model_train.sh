model=$1
python train.py --frame_features \
	--model=${model} \
	--feature_names="rgb,audio" \
	--feature_sizes="1024,128" \
	--train_data_pattern=/home/g0931848058johnny/yt8m/2/frame/train/train*.tfrecord \
	--train_dir="/home/g0931848058johnny/yt8m/models/frame/${model}" \
	--batch_size=32