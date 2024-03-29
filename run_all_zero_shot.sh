output_dir=zero_shot_zh_td_full
size="large-v2"
mkdir -p outputs/$output_dir
python3 zero_shot_tag_zh.py \
--batch 1 \
--output_dir outputs/$output_dir \
--custom_set_train data/TD/train.csv \
--custom_set_test data/TD/test.csv \
--size $size > outputs/$output_dir/output.log

output_dir=zero_shot_mix_td_full
size="large-v2"
mkdir -p outputs/$output_dir
python3 zero_shot_tag.py \
--batch 1 \
--output_dir outputs/$output_dir \
--custom_set_train data/TD/train.csv \
--custom_set_test data/TD/test.csv \
--load_cache \
--size $size > outputs/$output_dir/output.log

output_dir=zero_shot_zh_tat_full
size="large-v2"
mkdir -p outputs/$output_dir
python3 zero_shot_tag_zh.py \
--batch 1 \
--output_dir outputs/$output_dir \
--custom_set_train data/TAT/train.csv \
--custom_set_test data/TAT/test.csv \
--size $size > outputs/$output_dir/output.log

output_dir=zero_shot_mix_tat_full
size="large-v2"
mkdir -p outputs/$output_dir
python3 zero_shot_tag.py \
--batch 1 \
--output_dir outputs/$output_dir \
--custom_set_train data/TAT/train.csv \
--custom_set_test data/TAT/test.csv \
--load_cache \
--size $size > outputs/$output_dir/output.log

