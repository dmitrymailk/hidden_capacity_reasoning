    # --tasks leaderboard_mmlu_pro \
# lm_eval --model hf \
#     --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype="bfloat16" \
#     --tasks leaderboard_gpqa \
#     --device cuda:0 \
#     --batch_size 1 \
#     --apply_chat_template True

# https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about
# no --apply_chat_template
# |           Tasks            |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |----------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |leaderboard_gpqa            |       |none  |      |acc_norm|↑  |0.2827|±  |0.0131|
# | - leaderboard_gpqa_diamond |      1|none  |     0|acc_norm|↑  |0.2879|±  |0.0323|
# | - leaderboard_gpqa_extended|      1|none  |     0|acc_norm|↑  |0.2766|±  |0.0192|
# | - leaderboard_gpqa_main    |      1|none  |     0|acc_norm|↑  |0.2879|±  |0.0214|

# |     Groups     |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |----------------|-------|------|------|--------|---|-----:|---|-----:|
# |leaderboard_gpqa|       |none  |      |acc_norm|↑  |0.2827|±  |0.0131|

# using --apply_chat_template
# |           Tasks            |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |----------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |leaderboard_gpqa            |       |none  |      |acc_norm|↑  |0.2466|±  |0.0125|
# | - leaderboard_gpqa_diamond |      1|none  |     0|acc_norm|↑  |0.1970|±  |0.0283|
# | - leaderboard_gpqa_extended|      1|none  |     0|acc_norm|↑  |0.2802|±  |0.0192|
# | - leaderboard_gpqa_main    |      1|none  |     0|acc_norm|↑  |0.2277|±  |0.0198|

# |     Groups     |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |----------------|-------|------|------|--------|---|-----:|---|-----:|
# |leaderboard_gpqa|       |none  |      |acc_norm|↑  |0.2466|±  |0.0125|

##################
# test open ai api
##################
    # --tasks gsm8k \
    # --tasks gpqa_diamond_generative_n_shot \
# lm_eval --model local-chat-completions \
#     --model_args model=facebook/opt-125m,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
#     --apply_chat_template
##############
# lm_eval --model hf \
    # --tasks gpqa_diamond_generative_n_shot \
    # --tasks gsm8k \
# dataset=openai_math
# dataset=hendrycks_math
dataset=leaderboard_ifeval
lm_eval --model local-chat-completions \
    --model_args model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=bfloat16,base_url=http://172.17.0.1:1337/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
    --tasks  $dataset \
    --device cuda:0 \
    --batch_size 8 \
    --apply_chat_template True \
    --output_path ./test/$dataset \
    --gen_kwargs max_new_tokens=8192,temperature=0.6,max_tokens=8192,max_length=8192,do_sample=True \
    --log_samples \
    --apply_chat_template
    # --num_fewshot 1 \
# hf (pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,dtype=bfloat16), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
# |            Tasks             |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |------------------------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gpqa_diamond_generative_n_shot|      2|flexible-extract|     0|exact_match|↑  |0.0657|±  |0.0176|
# |                              |       |strict-match    |     0|exact_match|↑  |0.0000|±  |0.0000|