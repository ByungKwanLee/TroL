python eval/mathvista/extract_answer.py \
--output_dir /path_to_dataset/eval_results \
--output_file TroL-7B_mathvista_results.json 

python eval/mathvista/calculate_score.py \
--output_dir /path_to_dataset/eval_results \
--output_file TroL-7B_mathvista_results.json  \
--score_file TroL-7B_mathvista_scores.json \
--gt_file /path_to_dataset/MathVista/annot_testmini.json \