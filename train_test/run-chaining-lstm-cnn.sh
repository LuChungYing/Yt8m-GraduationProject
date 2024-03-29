CUDA_VISIBLE_DEVICES=0 python2 train.py \
  --train_dir="../model/lstm_cnn_deep_combine_chain/" \
  --train_data_pattern="/home/g0931848058johnny/yt8m/2/frame/train/*.tfrecord" \
  --frame_features=True \
  --feature_names="rgb,audio" \
  --feature_sizes="1024,128" \
  --model=LstmCnnDeepCombineChainModel \
  --deep_chain_layers=3 \
  --deep_chain_relu_cells=128 \
  --moe_num_mixtures=4 \
  --lstm_layers=1 \
  --lstm_cells="1024,128" \
  --rnn_swap_memory=True \
  --multitask=True \
  --label_loss=MultiTaskCrossEntropyLoss \
  --support_type="label,label,label" \
  --support_loss_percent=0.05 \
  --keep_checkpoint_every_n_hours=1.0 \
  --batch_size=96  \
  --num_readers=4 \
  --base_learning_rate=0.001

