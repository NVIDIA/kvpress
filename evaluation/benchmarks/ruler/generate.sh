# The following script prepares the synthetic data benchmark for a given Hugging Face tokenizer and without template
# Before running this script, make sure you downloaded the data as explained in the README:
# cd scripts/data/synthetic/json/
# python download_paulgraham_essay.py
# bash download_qa_dataset.sh

DATA_DIR="data/data"
TOKENIZER_PATH="meta-llama/Meta-Llama-3.1-8B"

SEQ_LENGTHS=(
    4096
    8192
    16384
)

TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    SAVE_DIR="${DATA_DIR}/${MAX_SEQ_LENGTH}"
    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir ${SAVE_DIR} \
            --benchmark synthetic \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type hf \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type base \
            --num_samples 500
    done
done