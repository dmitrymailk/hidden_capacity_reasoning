import os

os.environ["WANDB_PROJECT"] = "multiple_tokens_prediction_math_500"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import Qwen2ForCausalLM, Qwen2Model, AutoTokenizer, BitsAndBytesConfig
import torch
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
)

from datasets import load_dataset
from tqdm import tqdm
from hidden_capacity_reasoning.utils import (
    generate_train_examples,
    pad_train_examples,
    tokenize_single_turn,
)
from datasets import Dataset, IterableDataset
import gc
import types

# need for auto SFTTrainer patch(possible increase speed)
from unsloth import is_bfloat16_supported
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from hidden_capacity_reasoning.utils import (
    EOS_TOKEN_ID,
    TEXT_TOKEN_ID,
    WINDOW_SIZE,
    VISION_START,
    VISION_END,
    find_all_linear_names_v3,
)

import time
from datetime import datetime


from hidden_capacity_reasoning.models import (
    Qwen2ForCausalLMCompressionV1,
    Qwen2ModelEmbedPoolerV1,
    Qwen2ForCausalLMCompressionV2,
    Qwen2ModelEmbedPoolerV2,
    Qwen2ForCausalLMCompressionV3,
)
from transformers import AutoModelForCausalLM

from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm.contrib.concurrent import process_map
from tqdm_joblib import tqdm_joblib
from lm_eval.tasks.hendrycks_math.utils import strip_string, remove_boxed, is_equiv
from hidden_capacity_reasoning.evaluation.math_500.utils import (
    dataset_answer_filter,
    model_answer_filter,
)
import datasets


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


BLOCK_SIZE = 16


def multiple_token_prediction_data_v1(
    tokenized_turn,
    block_size=None,
):
    for key in tokenized_turn.keys():
        tokenized_turn[key] = torch.tensor(tokenized_turn[key])

    content_compression_mask = tokenized_turn["content_compression_mask"]

    input_part_end = (content_compression_mask == 0).nonzero()[-3][0]

    think_block_size = tokenized_turn["input_ids"].shape[0] - input_part_end - 1
    block_mask = torch.ones(think_block_size)

    original_input_ids = tokenized_turn["input_ids"].clone()
    tokens_dataset = []
    for block_pos in range(1, block_mask.shape[0] // block_size + 2):
        train_block_size = block_pos * block_size
        # temp_mask_1 = torch.ones(think_block_size)
        temp_block_mask = block_mask.clone()
        temp_block_mask[train_block_size:] = block_size

        if block_pos > 1:
            temp_block_mask[: (block_pos - 1) * block_size] = block_size

        temp_input_ids = original_input_ids.clone()
        temp_input_ids[input_part_end + 1 : input_part_end + 1 + think_block_size][
            temp_block_mask != block_size
        ] = TEXT_TOKEN_ID

        labels = tokenized_turn["input_ids"].clone()
        labels[content_compression_mask == 1] = TEXT_TOKEN_ID
        tokens_dataset.append(
            {
                "input_ids": temp_input_ids.tolist(),
                "labels": labels.tolist(),
            }
        )
    return tokens_dataset


def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(
        "dim/hendrycks_math_train_1k_DeepSeek-R1-Distill-Qwen-1.5B_max_len_4096_greedy"
    )

    base_prompt = open(
        "/code/hidden_capacity_reasoning/evaluation/math_500/math_500_prompt"
    ).read()

    dataset = dataset["train"].train_test_split(
        test_size=350,
        # test_size=1,
        seed=42,
    )
    dataset = dataset["test"].filter(lambda x: x["model_answer"].count("</think>") == 1)

    dataset = dataset.rename_columns(
        {
            "problem": "question",
            "answer": "gold_answer",
            "model_answer": "answer",
        }
    )
    dataset = dataset.remove_columns(
        [item for item in dataset.column_names if not item in ["question", "answer"]]
    )

    # test pass
    tokenize_single_turn(
        question=base_prompt.format(question=dataset[0]["question"]),
        answer=dataset[0]["answer"],
        tokenizer=tokenizer,
    )
    train_examples = [
        tokenize_single_turn(
            tokenizer=tokenizer,
            question=base_prompt.format(question=item["question"]),
            answer=item["answer"],
        )
        for item in tqdm(dataset.to_list())
        # for item in tqdm(dataset.to_list()[:2000])
        # for item in tqdm(dataset.to_list()[:1])
    ]

    prepared_train_examples = []
    with tqdm_joblib(
        tqdm(desc="My calculation", total=len(train_examples))
    ) as progress_bar:
        examples = Parallel(n_jobs=-1)(
            delayed(multiple_token_prediction_data_v1)(
                tokenized_turn=tokenized_turn,
                block_size=BLOCK_SIZE,
            )
            for tokenized_turn in train_examples
        )
    for example in examples:
        for item in example:
            prepared_train_examples.append(item)

    # new_dataset = Dataset.from_list(prepared_train_examples)
    new_dataset = CustomDataset(prepared_train_examples)
    print(dataset)

    def collate_fn(batch):
        # только для batch=1
        inputs_str = [tokenizer.decode(item["input_ids"]) for item in batch]
        labels = [tokenizer.decode(item["labels"]) for item in batch]
        # batch
        padded_batch = tokenizer(
            inputs_str,
            padding="longest",
            return_tensors="pt",
        )
        labels = tokenizer(
            labels,
            padding="longest",
            return_tensors="pt",
        )
        padded_batch["labels"] = labels["input_ids"]

        skip_ids = [
            TEXT_TOKEN_ID,
            EOS_TOKEN_ID,
        ]
        for skip_id in skip_ids:
            padded_batch["labels"][padded_batch["labels"] == skip_id] = -100

        return padded_batch

    peft_config = LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            # "out_proj",
        ],
        modules_to_save=[
            "embed_tokens",
        ],
    )

    formatted_date = datetime.fromtimestamp(time.time()).strftime(
        "%Y_%m_%d_%H_%M_%S_%f"
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=new_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            # gradient_accumulation_steps=4,
            warmup_steps=1,
            # num_train_epochs=1,  # 90,  # Set this for 1 full training run.
            num_train_epochs=2,  # Set this for 1 full training run.
            # max_steps=10000,
            learning_rate=1e-4,
            # bf16=model.dtype == torch.bfloat16,
            bf16=True,
            # fp16=model.dtype == torch.float16,
            logging_steps=8,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=f"multiple_tokens/{formatted_date}",
            report_to="wandb",
            # report_to="none",
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            gradient_checkpointing=True,
            save_steps=5000,
            run_name=formatted_date,
        ),
    )

    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Requires Gradient: {param.requires_grad}")
    trainer.train()


if __name__ == "__main__":
    main()
