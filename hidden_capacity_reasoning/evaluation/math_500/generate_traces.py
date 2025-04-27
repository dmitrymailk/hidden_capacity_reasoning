from more_itertools import chunked
from tqdm import tqdm
import json

import concurrent
import openai
import logging
from datasets import Dataset

logging.getLogger("openai").setLevel(logging.ERROR)

MAX_TOKENS = 4096

# https://docs.together.ai/docs/prompting-deepseek-r1
def sglang_generate(prompt: str):

    client = openai.Client(
        base_url=f"http://172.17.0.1:1337/v1",
        api_key="None",
    )
    response = client.chat.completions.create(
        model="sglang",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        # temperature=0.6,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        # top_p=0.95,
    )

    model_responce = response.choices[0].message.content
    return model_responce


def batch_generation_sglang(prompts):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        prompts_results = list(
            executor.map(
                sglang_generate,
                prompts,
            )
        )
    return prompts_results

if __name__ == "__main__":

    dataset = open(
        "hidden_capacity_reasoning/evaluation/math_500/datasets/test.jsonl"
    ).readlines()
    
    dataset = [json.loads(item) for item in dataset]

    base_prompt = open(
        "hidden_capacity_reasoning/evaluation/math_500/math_500_prompt"
    ).read()

    batch_size = 128 * 2
    new_dataset = dataset
    dataset_with_answers = []
    
    for batch in tqdm(list(chunked(new_dataset, batch_size))):
        test_problems = [item["problem"] for item in batch]
        test_problems = [base_prompt.format(question=item) for item in test_problems]
        test_problems = batch_generation_sglang(test_problems)
        for answer, item in zip(test_problems, batch):
            item["model_answer"] = answer

        dataset_with_answers.extend(batch)
    
    dataset_with_answers = Dataset.from_list(dataset_with_answers)
    # dataset_with_answers.push_to_hub(f'dim/hendrycks_math_train_12k_DeepSeek-R1-Distill-Qwen-1.5B_max_len_{MAX_TOKENS}')
    dataset_with_answers.push_to_hub(f'dim/hendrycks_math_test_500_DeepSeek-R1-Distill-Qwen-1.5B_max_len_{MAX_TOKENS}_greedy')
    