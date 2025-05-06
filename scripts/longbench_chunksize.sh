cd evaluation/LongBench

model="longchat-v1.5-7b-32k"

# python -u pred.py \
#     --model $model --task qasper

for task in "qasper" "multifieldqa_en" "narrativeqa" "hotpotqa" "triviaqa"
do
  for chunk_size in 24
  # for chunk_size in 1 2 4 8 16 32 64 128 256
  do
      python -u pred.py \
          --model $model --task $task \
          --quest --token_budget 4096 --chunk_size $chunk_size
  done
done
# python -u eval.py --model $model