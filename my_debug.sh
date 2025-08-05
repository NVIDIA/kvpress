pip install .
cd evaluation/
python evaluate.py --press_name sepllm_trnfree --dataset ruler --model meta-llama/Meta-Llama-3.1-8B-Instruct --compression_ratio 0.75 2>&1 | tee tmp.log
cd -