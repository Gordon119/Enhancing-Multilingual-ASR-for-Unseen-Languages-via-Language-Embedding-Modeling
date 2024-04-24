output_dir=$1
size=$2
mkdir -p outputs/$output_dir
python3 train_peft_TAT_TD_mix.py --grad_accum 8 --batch 1 --specified_epoch 0 --total_epoch 5 --output_dir outputs/$output_dir --size $size > outputs/$output_dir/output.log
