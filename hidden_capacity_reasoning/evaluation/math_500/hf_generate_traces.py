from more_itertools import chunked
from tqdm import tqdm
import json

import logging
from datasets import Dataset

logging.getLogger("openai").setLevel(logging.ERROR)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.set_grad_enabled(False)
MAX_TOKENS = 4096


if __name__ == "__main__":

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        torch_dtype=torch.float32,
        # torch_dtype=torch.bfloat16,
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = open(
        # "hidden_capacity_reasoning/evaluation/math_500/datasets/test.jsonl"
        "hidden_capacity_reasoning/evaluation/math_500/datasets/train.jsonl"
    ).readlines()[:1000]

    dataset = [json.loads(item) for item in dataset]

    base_prompt = open(
        "hidden_capacity_reasoning/evaluation/math_500/math_500_prompt"
    ).read()

    batch_size = 4
    new_dataset = dataset
    dataset_with_answers = []
    device = "cuda"

    for batch in tqdm(list(chunked(new_dataset, batch_size))):
        test_problems = [item["problem"] for item in batch]
        test_problems = [base_prompt.format(question=item) for item in test_problems]
        test_problems = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": item},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in test_problems
        ]

        model_inputs = tokenizer(
            test_problems,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            add_special_tokens=False,
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=4096,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # test_problems = batch_generation_sglang(test_problems)
        for answer, item in zip(responses, batch):
            item["model_answer"] = answer

        dataset_with_answers.extend(batch)

    dataset_with_answers = Dataset.from_list(dataset_with_answers)
    dataset_with_answers.push_to_hub(f'dim/hendrycks_math_train_1k_DeepSeek-R1-Distill-Qwen-1.5B_max_len_{MAX_TOKENS}_greedy')
    # dataset_with_answers.push_to_hub(
    #     f"dim/hendrycks_math_test_500_DeepSeek-R1-Distill-Qwen-1.5B_max_len_{MAX_TOKENS}_greedy"
    # )
