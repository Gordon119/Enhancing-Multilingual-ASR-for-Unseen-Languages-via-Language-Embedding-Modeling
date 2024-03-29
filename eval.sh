output_dir=$1
size=$2
mkdir -p outputs/$output_dir
python3 eval_peft_TAT_TD.py --load_cache --grad_accum 8 --batch 2 --specified_epoch 1 --total_epoch 1 --output_dir outputs/$output_dir --size $size --only_eval --checkpoint Gordon119/TAT-openai-whisper-large-v2-mix-tag-epoch2-total5epoch > outputs/$output_dir/output.log
