model=$1

output_dir="/home/g0931848058johnny/yt8m/models/model_predictions/train/${model}"
      CUDA_VISIBLE_DEVICES="$GPU_ID" python new_inference-pre-ensemble.py \
        --output_dir=$output_dir \
        --train_dir="/home/g0931848058johnny/yt8m/models/frame/${model}" \
        --input_data_pattern="/home/g0931848058johnny/yt8m/3/frame/validate/*.tfrecord" \
        --segment_labels \
        --model=${model} \
        --frame_features=True \
        --feature_names="rgb,audio" \
        --feature_sizes="1024,128" \
        --batch_size=32 \
        --file_size=4096
        

     
        
