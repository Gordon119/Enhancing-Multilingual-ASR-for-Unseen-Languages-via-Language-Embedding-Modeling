set -e
seed=$1
# utterance wise
exp_name="finetune_utterance_wise"
for dir in data/ml_superb/sixth_edition/languages/all;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$exp_name/$lang
    if [ -d "outputs/$exp_name/$lang" ]; then
        continue
    else
        mkdir -p $output_dir
    fi
    if [ $lang == "all" ]; then
        python3 ws_finetune_untrainable.py \
            --size $size \
            --batch 1 \
            --grad_accum 8 \
            --epoch 5 \
            --all \
            --seed $seed \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --output_dir $output_dir > $output_dir/output.log
    else
        python3 ws_finetune_untrainable.py \
            --size $size \
            --batch 1 \
            --grad_accum 8 \
            --epoch 5 \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --output_dir $output_dir > $output_dir/output.log
    fi
done

# corpus wise
exp_name="finetune_corpus_wise"
for dir in data/ml_superb/sixth_edition/languages/all;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$exp_name/$lang
    if [ -d "outputs/$exp_name/$lang" ]; then
        continue
    else
        mkdir -p $output_dir
    fi
    if [ $lang == "all" ]; then
        python3 ws_finetune_untrainable.py \
            --size $size \
            --batch 1 \
            --grad_accum 8 \
            --epoch 5 \
            --all \
            --seed $seed \
            --corpus_wise \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --output_dir $output_dir > $output_dir/output.log
    else
        python3 ws_finetune_untrainable.py \
            --size $size \
            --batch 1 \
            --grad_accum 8 \
            --epoch 5 \
            --corpus_wise \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --output_dir $output_dir > $output_dir/output.log
    fi
done
