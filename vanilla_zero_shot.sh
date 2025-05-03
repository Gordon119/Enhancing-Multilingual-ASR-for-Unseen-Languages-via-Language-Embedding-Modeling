seed=$1
exp_name="zero_shot_vanilla"
for dir in data/ml_superb/sixth_edition/languages/all;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$seed/$exp_name/$lang
    mkdir -p $output_dir
    python3 vanilla_zero_shot.py \
        --size $size \
        --batch 1 \
        --seed $seed \
        --custom_set_train $dir/train.csv \
        --custom_set_test $dir/test_val.csv \
        --output_dir $output_dir > $output_dir/output.log
done
