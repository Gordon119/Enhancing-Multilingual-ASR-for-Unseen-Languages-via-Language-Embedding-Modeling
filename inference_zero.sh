size="large-v2"
for top_k in {1..99};do
    for lang in ceb zul xho gle kam nob umb pus nya nep ful nso ibo ori fil kea lug ast ckb pan orm oci kir wol luo msa mya yue ell swa ven nbl ssw tsn sot tso;do
        output_dir=outputs/top_$top_k/$lang
        mkdir -p $output_dir
        repo_name=/home/gordon1109/Whisper_Experiments/data/ml_superb/sixth_edition/$lang
        python3 zero_shot.py \
            --size $size \
            --batch 2 \
            --grad_accum 4 \
            --custom_set_train $repo_name/train.csv \
            --custom_set_test $repo_name/test_val.csv \
            --repo_name $repo_name \
            --output_dir $output_dir \
            --top_k $top_k \
            --only_eval > $output_dir/output.log    
    done
done
