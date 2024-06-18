# OpenAI Key
OPENAI_KEY = ""

# WANDB Key
WANDB_KEY = ""
WANDB_ID = ""

# Checkpoints & Dataset root
DATASET_ROOT=""
TROL_1_8B="BK-Lee/TroL-1.8B"
TROL_3_8B="BK-Lee/TroL-3.8B"
TROL_7B="BK-Lee/TroL-7B"

# Json files for Evaluation
VQAV2 = "VQAv2/v2_OpenEnded_mscoco_test2015_questions.json"
GQA = "gqa/testdev_balanced_questions.json"
SQA = "ScienceQA/problems.json"
SQA_SPLIT = "ScienceQA/pid_splits.json"
VIZWIZ = "VizWiz/test.json"
TEXTVQA = "TextVQA/llava_textvqa_val_v051_ocr.json"
TEXTVQA_ANNOTATIONS = "TextVQA/TextVQA_0.5.1_val.json"
POPE_POPULAR = "POPE/coco_pope_popular.json"
POPE_ADVERSARIAL = "POPE/coco_pope_adversarial.json"
POPE_RANDOM = "POPE/coco_pope_random.json"
MME = "MME_Benchmark_release_version/llava_mme.json"
MME_DIR = "MME_Benchmark_release_version"
MMBENCH = "MMBench/MMBench_TEST_EN_legacy.tsv"
MMBENCH_CN = "MMBench/MMBench_TEST_CN_legacy.tsv"
MMBENCH_DEV = "MMBench/mmbench_dev_20230712.tsv"
MMBENCH_CN_DEV = "MMBench/mmbench_dev_cn_20231003.tsv"
QBENCH = "LLVisionQA-QBench/llvisionqa_dev.json"
QBENCH_CN = "LLVisionQA-QBench/质衡-问答-验证集.json"
MMVET = "mm-vet/mm-vet.json"
MMMU = "MMMU/*/validation*"
MATHVISTA = "MathVista/testmini-00000-of-00001-725687bf7a18d64b.parquet"
AI2D = "ai2d/ai2d_test.json"
HALLUSIONBENCH = "HallusionBench/HallusionBench.json"
CHARTQA = "chartqa/test/test_augmented.json"
SEED = "SEED-Bench/SEED-Bench.json"
LLAVA = "llava-bench-in-the-wild/questions.jsonl"
MMSTAR = "MMStar/mmstar.parquet"
MATHVERSE = "MathVerse/testmini.json"
MATHVERSE_TEXT_ONLY = "MathVerse/testmini_text_only.json"
VISUALWEBBENCH = "VisualWebBench/*"

# Available evaluation datasets
EVAL_DATASETS = ["qbench", "sqa", "ai2d", "chartqa", "seed", "pope", "hallusionbench", "mme", \
                 "mathvista", "mmbench", "mmbench_cn", "mmvet", "llava", "mmstar", "mathverse", "visualwebbench"]

# Download number 
DOWNLOAD_NUMBER = 5
