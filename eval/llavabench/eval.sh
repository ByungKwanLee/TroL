python eval/llavabench/eval_gpt_review_bench.py \
    --question /path_to_dataset/llava-bench-in-the-wild/questions.jsonl \
    --context /path_to_dataset/llava-bench-in-the-wild/context.jsonl \
    --rule eval/llavabench/rule.json \
    --answer-list \
        /path_to_dataset/llava-bench-in-the-wild/answers_gpt4.jsonl \
        /path_to_dataset/eval_results/TroL-7B_llava_results.jsonl \
    --output \
        /path_to_dataset/eval_results/reviews_trol_llava_results.jsonl

python eval/llavabench/summarize_gpt_review.py -f /path_to_dataset/eval_results/reviews_trol-7b_llava_results.jsonl
