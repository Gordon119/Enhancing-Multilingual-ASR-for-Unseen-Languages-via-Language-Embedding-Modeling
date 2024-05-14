# utterance wise
exp_name=$1
for dir in data/ml_superb/sixth_edition/languages/*;do
    size=large-v2
    lang=$(basename $dir)
    if [ $lang == "all" ]; then
        continue
    fi
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    repo_name=$dir
    python3 ws_inference.py \
        --size $size \
        --custom_set_train $repo_name/train.csv \
        --custom_set_test $repo_name/test_val.csv \
        --repo_name $repo_name \
        --output_dir $output_dir > $output_dir/output.log
done

# corpus wise
# exp_name=$1
# for dir in data/ml_superb/sixth_edition/languages/*;do
#     size=large-v2
#     lang=$(basename $dir)
#     if [ $lang == "all" ]; then
#         continue
#     fi
#     output_dir=outputs/$exp_name/$lang
#     mkdir -p $output_dir
#     repo_name=$dir
#     python3 ws_inference.py \
#         --size $size \
#         --custom_set_train $repo_name/train.csv \
#         --custom_set_test $repo_name/test_val.csv \
#         --repo_name $repo_name \
#         --corpus_wise \
#         --output_dir $output_dir > $output_dir/output.log
# done
