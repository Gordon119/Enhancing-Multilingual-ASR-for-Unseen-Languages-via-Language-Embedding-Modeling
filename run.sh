set -e
for file in vanilla_zero_shot_seen ws_zero_shot_mlp; do
    bash "$file".sh $seed
done
