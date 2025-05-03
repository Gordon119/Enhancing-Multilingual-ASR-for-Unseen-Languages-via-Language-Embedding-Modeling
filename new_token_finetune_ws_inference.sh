set -e
seed=$1
exp_name="finetune_vanilla_utterance_wise"
for dir in data/ml_superb/sixth_edition/languages/all;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$seed/$exp_name/$lang
    if [ -d "outputs/$exp_name/$lang" ]; then
        continue
    else
        mkdir -p $output_dir
    fi
    if [ $lang == "all" ]; then
        python3 new_token_finetune_ws_inference.py \
            --size $size \
            --batch 1 \
            --seed $seed \
            --grad_accum 8 \
            --epoch 5 \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --all \
            --output_dir $output_dir > $output_dir/output.log
    else
        python3 new_token_finetune_ws_inference.py \
        --size $size \
        --batch 1 \
        --seed $seed \
        --grad_accum 8 \
        --epoch 5 \
        --custom_set_train $dir/train.csv \
        --custom_set_test $dir/test_val.csv \
        --output_dir $output_dir > $output_dir/output.log
    fi
done

exp_name="finetune_vanilla_corpus_wise"
for dir in data/ml_superb/sixth_edition/languages/all;do
    size=large-v2
    lang=$(basename $dir)
    output_dir=outputs/$seed/$exp_name/$lang
    if [ -d "outputs/$exp_name/$lang" ]; then
        continue
    else
        mkdir -p $output_dir
    fi
    if [ $lang == "all" ]; then
        python3 new_token_finetune_ws_inference.py \
            --size $size \
            --batch 1 \
            --seed $seed \
            --grad_accum 8 \
            --epoch 5 \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --all \
            --corpus_wise \
            --output_dir $output_dir > $output_dir/output.log
    else
        python3 new_token_finetune_ws_inference.py \
            --size $size \
            --batch 1 \
            --seed $seed \
            --grad_accum 8 \
            --epoch 5 \
            --custom_set_train $dir/train.csv \
            --custom_set_test $dir/test_val.csv \
            --corpus_wise \
            --output_dir $output_dir > $output_dir/output.log
    fi
done
