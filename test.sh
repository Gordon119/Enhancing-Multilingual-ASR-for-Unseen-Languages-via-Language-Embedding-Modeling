for dir in data/ml_superb/sixth_edition/languages/*;do
    size=large-v2
    lang=$(basename $dir)
    if [ $lang == "all" ]; then
        continue
    fi
    output_dir=outputs/all_fix_finetune_untrainable/$lang
    mkdir -p $output_dir
    repo_name=$dir
    python3 zero_shot_untrainable.py \
        --size $size \
        --batch 1 \
        --grad_accum 8 \
        --epoch 5 \
        --custom_set_train $repo_name/train.csv \
        --custom_set_test $repo_name/test_val.csv \
        --repo_name $repo_name \
        --test_seperate \
        --output_dir $output_dir > $output_dir/output.log
done
