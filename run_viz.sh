#!/bin/bash

python viz.py metric --result_dir test_results/meta-llama@Llama-2-7b-chat-hf__mmlu --out_dir plots_mmlu
python viz.py metric --result_dir test_results/meta-llama@Llama-2-7b-chat-hf__rotten_tomatoes --out_dir plots_rotten_tomatoes
python viz.py metric --result_dir test_results/meta-llama@Llama-2-7b-chat-hf__tweetqa --out_dir plots_tweetqa