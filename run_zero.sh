set -e
size=large-v2
for topk in {45..98};do
    for file in data/ml_superb/sixth_edition/*_all.csv;do
        dir=$(basename -- $file)
        lang="${dir%_all.*}"
        mkdir -p outputs/topk_$topk/${lang}_all
        CUDA_VISIBLE_DEVICES=0 python3 zero_shot_tag.py --batch 1 --specified_epoch 0 --total_epoch 5 --custom_set_test $file --output_dir outputs/topk_$topk/${lang}_all --size $size --top_k $topk --get_weight > outputs/topk_$topk/${lang}_all/output.log
    done
done 
