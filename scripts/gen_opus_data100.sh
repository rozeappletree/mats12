cd /root/mats12
setsid nohup python -u scripts/gen_opus_data100.py \
    --per_level 100 \
    --output_dir datasets_claudeopus_sample2 \
    > datasets_claudeopus_sample2/run.log 2>&1 &
echo $! > datasets_claudeopus_sample2/run.pid
