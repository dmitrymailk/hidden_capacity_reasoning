import os

os.environ["WANDB_PROJECT"] = "hidden_capacity_reasoning"
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
from datasets import Dataset
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
)


def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = Qwen2ForCausalLMCompressionV2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.model.requires_grad_(False)

    temp_model = Qwen2ModelEmbedPoolerV2.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        # quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    )
    print(model.embed_pooler.load_state_dict(temp_model.state_dict()))
    temp_model = temp_model.cpu()
    del temp_model
    gc.collect()
    torch.cuda.empty_cache()

    dataset = load_dataset("dim/open_orca_905_DeepSeek-R1-Distill-Qwen-1.5B")
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=10, seed=42)

    # test pass
    tokenize_single_turn(
        question=dataset["train"][0]["question"],
        answer=dataset["train"][0]["answer"],
        tokenizer=tokenizer,
    )
    train_examples = [
        tokenize_single_turn(tokenizer=tokenizer, **item)
        for item in tqdm(dataset["train"].to_list()[:3])
    ]

    prepared_train_examples = []
    for item in tqdm(train_examples):
        for example in generate_train_examples(
            dataset_batch=[item],
            window_size=WINDOW_SIZE,
        ):
            prepared_train_examples.append(example)

    print(
        "max_len",
        max([len(item["original_tokens"]) for item in prepared_train_examples]),
    )

    new_dataset = Dataset.from_list(prepared_train_examples)
    print(dataset)

    def collate_fn(batch):
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
        # print(padded_batch)
        return padded_batch

    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=find_all_linear_names_v3(model=model),
        modules_to_save=["embed_pooler.model.embed_tokens"],
    )

    formatted_date = datetime.fromtimestamp(time.time()).strftime(
        "%Y_%m_%d_%H_%M_%S_%f"
    )
    model.embed_pooler = prepare_model_for_kbit_training(model.embed_pooler)
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
            warmup_steps=5,
            # num_train_epochs=1,#90,  # Set this for 1 full training run.
            num_train_epochs=90,  # Set this for 1 full training run.
            # max_steps=10000,
            learning_rate=1e-4,
            bf16=model.dtype == torch.bfloat16,
            fp16=model.dtype == torch.float16,
            logging_steps=8,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=f"outputs/{formatted_date}",
            # report_to="wandb",
            report_to="none",
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            # gradient_checkpointing=True,
            save_steps=10000,
            run_name=formatted_date,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
