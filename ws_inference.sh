# utterance wise
exp_name="inference_utterance_wise"
for dir in data/ml_superb/sixth_edition/languages/mya data/ml_superb/sixth_edition/languages/ori data/ml_superb/sixth_edition/languages/pan;do
    size=large-v2
    lang=$(basename $dir)
    if [ $lang == "all" ]; then
        continue
    fi
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    CUDA_VISIBLE_DEVICES=1 python3 ws_inference.py \
        --size $size \
        --batch 1 \
        --custom_set_test $dir/test_val.csv \
        --output_dir $output_dir > $output_dir/output.log
done

# corpus wise
exp_name="inference_corpus_wise"
for dir in data/ml_superb/sixth_edition/languages/mya data/ml_superb/sixth_edition/languages/ori data/ml_superb/sixth_edition/languages/pan;do
    size=large-v2
    lang=$(basename $dir)
    if [ $lang == "all" ]; then
        continue
    fi
    output_dir=outputs/$exp_name/$lang
    mkdir -p $output_dir
    python3 ws_inference.py \
        --size $size \
        --batch 1 \
        --custom_set_test $dir/test_val.csv \
        --corpus_wise \
        --output_dir $output_dir > $output_dir/output.log
done
