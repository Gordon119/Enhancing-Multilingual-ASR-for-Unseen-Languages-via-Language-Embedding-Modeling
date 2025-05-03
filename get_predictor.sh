set -e
for dir in data/ml_superb/sixth_edition/seen_languages/*;do
    size=large-v2
    lang=$(basename $dir)
    python3 get_weight.py \
        --size $size \
        --batch 1 \
        --mask $lang \
        --custom_set_train $dir/train.csv
done

# Train the predictor
python embedding_predictor.py
