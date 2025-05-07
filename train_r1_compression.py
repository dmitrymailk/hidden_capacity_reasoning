import os

os.environ["WANDB_PROJECT"] = "hidden_capacity_reasoning_math_500"
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


def main():
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = "r1_compressor_v2"
    model = Qwen2ForCausalLMCompressionV2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.model.requires_grad_(False)
    model.lm_head.requires_grad_(False)

    # dataset = load_dataset("dim/open_orca_4475_DeepSeek-R1-Distill-Qwen-1.5B")
    # dataset = dataset["train"]
    # dataset = dataset.train_test_split(test_size=500, seed=42)
    # dataset = load_dataset("dim/hendrycks_math_train_12k_DeepSeek-R1-Distill-Qwen-1.5B_max_len_4096")
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
    # correct_dataset = []

    # for pos, item in enumerate(dataset):
    #     try:
    #         answer = dataset_answer_filter(item["answer"])
    #         model_answer = model_answer_filter(item["model_answer"])
    #         # print(answer, model_answer)
    #         # break
    #         if is_equiv(answer, model_answer):
    #             correct_dataset.append(item)
    #     except:
    #         pass
    # dataset = datasets.Dataset.from_list(correct_dataset)
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
            delayed(generate_train_examples)(
                dataset_batch=[item], window_size=WINDOW_SIZE
            )
            for item in train_examples
        )
    for example in examples:
        for item in example:
            prepared_train_examples.append(item)

    print(
        "max_len",
        max([len(item["original_tokens"]) for item in prepared_train_examples]),
    )

    # new_dataset = Dataset.from_list(prepared_train_examples)
    new_dataset = CustomDataset(prepared_train_examples)
    print(dataset)

    def collate_fn(batch):
        # только для batch=1
        padded_batch = pad_train_examples(
            train_examples=batch,
            tokenizer=tokenizer,
        )
        padded_batch = {
            "replaced_original_tokens": padded_batch["replaced_original_tokens"][
                "input_ids"
            ],
            "compressed_input_ids": padded_batch["compressed_input_ids"]["input_ids"],
            "original_tokens": padded_batch["original_tokens"]["input_ids"],
            "attention_mask": padded_batch["compressed_input_ids"]["attention_mask"],
            "labels": padded_batch["compressed_input_ids"]["input_ids"],
            "content_compression_mask": padded_batch["content_compression_mask"][
                "input_ids"
            ],
        }
        for key in padded_batch.keys():
            padded_batch[key] = torch.tensor(padded_batch[key])
        skip_ids = [
            TEXT_TOKEN_ID,
            EOS_TOKEN_ID,
            VISION_START,
            VISION_END,
        ]
        for skip_id in skip_ids:
            padded_batch["labels"][padded_batch["labels"] == skip_id] = -100
        # часть инпута от пользователя
        last_index = (
            (padded_batch["content_compression_mask"] == 1).long().nonzero()[-1][1]
        )
        # print("======")
        # print((padded_batch["content_compression_mask"] == 1).long().nonzero())
        # print((padded_batch["content_compression_mask"] == 1).long().nonzero()[-1])
        # print((padded_batch["content_compression_mask"] == 1).long().nonzero()[-1][1])
        # print("======")
        # print("======")
        # print("======")
        # print("======")
        padded_batch["labels"][:, : last_index + 1][
            padded_batch["content_compression_mask"][:, : last_index + 1] == 1
        ] = -100
        # print(padded_batch)
        return padded_batch

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=find_all_linear_names_v3(model=model),
        modules_to_save=[
            "embed_pooler.model.embed_tokens",
            "embed_pooler.weight_pooler",
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
            gradient_accumulation_steps=1,
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
            output_dir=f"outputs/{formatted_date}",
            report_to="wandb",
            # report_to="none",
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            gradient_checkpointing=True,
            save_steps=10000,
            run_name=formatted_date,
        ),
    )

    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Requires Gradient: {param.requires_grad}")
    trainer.train()


if __name__ == "__main__":
    main()
