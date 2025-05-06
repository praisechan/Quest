cd evaluation/pg19
device=0
budget=4096
MODELPATH=lmsys/longchat-7b-v1.5-32k

# OUTPUT_DIR=results/ppl_eval/longchat_1
# mkdir -p $OUTPUT_DIR
# python -u ppl_eval.py --model_name_or_path $MODELPATH --output_dir $OUTPUT_DIR --num_eval_tokens 30000 --quest --token_budget $budget --chunk_size 1
OUTPUT_DIR=results/ppl_eval/longchat_2
mkdir -p $OUTPUT_DIR
python -u ppl_eval.py --model_name_or_path $MODELPATH --output_dir $OUTPUT_DIR --num_eval_tokens 30000 --quest --token_budget $budget --chunk_size 2
# OUTPUT_DIR=results/ppl_eval/longchat_4
# mkdir -p $OUTPUT_DIR
# python -u ppl_eval.py --model_name_or_path $MODELPATH --output_dir $OUTPUT_DIR --num_eval_tokens 30000 --quest --token_budget $budget --chunk_size 4
# OUTPUT_DIR=results/ppl_eval/longchat_8
# mkdir -p $OUTPUT_DIR
# python -u ppl_eval.py --model_name_or_path $MODELPATH --output_dir $OUTPUT_DIR --num_eval_tokens 30000 --quest --token_budget $budget --chunk_size 8
# OUTPUT_DIR=results/ppl_eval/longchat_16
# mkdir -p $OUTPUT_DIR
# python -u ppl_eval.py --model_name_or_path $MODELPATH --output_dir $OUTPUT_DIR --num_eval_tokens 30000 --quest --token_budget $budget --chunk_size 16
# OUTPUT_DIR=results/ppl_eval/longchat_32
# mkdir -p $OUTPUT_DIR
# python -u ppl_eval.py --model_name_or_path $MODELPATH --output_dir $OUTPUT_DIR --num_eval_tokens 30000 --quest --token_budget $budget --chunk_size 32
# OUTPUT_DIR=results/ppl_eval/longchat_64
# mkdir -p $OUTPUT_DIR
# python -u ppl_eval.py --model_name_or_path $MODELPATH --output_dir $OUTPUT_DIR --num_eval_tokens 30000 --quest --token_budget $budget --chunk_size 64
# OUTPUT_DIR=results/ppl_eval/longchat_128
# mkdir -p $OUTPUT_DIR
# python -u ppl_eval.py --model_name_or_path $MODELPATH --output_dir $OUTPUT_DIR --num_eval_tokens 30000 --quest --token_budget $budget --chunk_size 128
# OUTPUT_DIR=results/ppl_eval/longchat_256
# mkdir -p $OUTPUT_DIR
# python -u ppl_eval.py --model_name_or_path $MODELPATH --output_dir $OUTPUT_DIR --num_eval_tokens 30000 --quest --token_budget $budget --chunk_size 256