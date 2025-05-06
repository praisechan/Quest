cd evaluation/LongBench

model="longchat-v1.5-7b-32k"

# for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
# for task in "qasper" "multifieldqa_en"
python -u pred.py \
    --model $model --task multifieldqa_en

for chunk_size in 1 2 4 8 16 32 64 128 256
do
    python -u pred.py \
        --model $model --task multifieldqa_en \
        --quest --token_budget 4096 --chunk_size $chunk_size
done

# python -u eval.py --model $model