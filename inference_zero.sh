size="large-v2"
for top_k in {1..99};do
    for dir in data/ml_superb/sixth_edition/languages/*;do
        lang=$(basename $dir)
        output_dir=outputs/top_$top_k/$lang
        mkdir -p $output_dir
        repo_name=$dir
        python3 zero_shot.py \
            --size $size \
            --batch 4 \
            --custom_set_train $repo_name/train.csv \
            --custom_set_test $repo_name/test_val.csv \
            --repo_name $repo_name \
            --output_dir $output_dir \
            --top_k $top_k \
            --only_eval > $output_dir/output.log    
    done
done
