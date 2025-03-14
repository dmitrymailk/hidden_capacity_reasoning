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
from hidden_capacity_reasoning.utils import EOS_TOKEN_ID, TEXT_TOKEN_ID, WINDOW_SIZE

import time
from datetime import datetime


class Qwen2ModelEmbedPoolerV1(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.model.embed_tokens = None
        self.lm_head = None
        self.post_init()

    def forward(self, input_embeds):
        # print(input_embeds.dtype)
        input_embeds = self.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )[0]
        # print(input_embeds.dtype)
        input_embeds = input_embeds.sum(1) / torch.tensor(
            input_embeds.shape[1],
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )
        # print(input_embeds.dtype)
        input_embeds = input_embeds.unsqueeze(1)
        return input_embeds


class Qwen2ForCausalLMCompressionV1(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        # print(config._name_or_path)
        self.embed_pooler = Qwen2ModelEmbedPoolerV1.from_pretrained(
            config._name_or_path,
            config=config,
            attn_implementation="flash_attention_2",
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        if "replaced_original_tokens" in kwargs:
            original_tokens_torch = kwargs["original_tokens"].to(self.model.device)
            replaced_tokens_torch = kwargs["replaced_original_tokens"].to(
                self.model.device
            )
            compressed_tokens_torch = kwargs["compressed_input_ids"].to(
                self.model.device
            )

            original_embeds = self.model.get_input_embeddings()(original_tokens_torch)
            compressed_embeds_template = self.model.get_input_embeddings()(
                compressed_tokens_torch
            )

            tokens_for_compression_mask = replaced_tokens_torch == TEXT_TOKEN_ID
            compressed_tokens_mask = compressed_tokens_torch == TEXT_TOKEN_ID
            embeds_for_compression = original_embeds[
                tokens_for_compression_mask
            ].reshape(
                -1,
                WINDOW_SIZE,
                original_embeds.shape[-1],
            )
            pooled_embeds = self.embed_pooler(embeds_for_compression)
            pooled_embeds = pooled_embeds.to(compressed_embeds_template.dtype)
            compressed_embeds_template = compressed_embeds_template.masked_scatter(
                compressed_tokens_mask.unsqueeze(-1).expand_as(
                    compressed_embeds_template
                ),
                pooled_embeds,
            )
            inputs_embeds = compressed_embeds_template
        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            cache_position,
            logits_to_keep,
            **kwargs,
        )


def find_all_linear_names_v2(model):
    lora_module_names = set()
    target_modules = set(
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "embed_pooler" in name:
                names = name.split(".")[-1]
                if names in target_modules:
                    lora_module_names.add(name)
    return lora_module_names


def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = Qwen2ForCausalLMCompressionV1.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.model.requires_grad_(False)

    temp_model = Qwen2ModelEmbedPoolerV1.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    print(model.embed_pooler.load_state_dict(temp_model.state_dict()))
    temp_model = temp_model.cpu()
    del temp_model
    gc.collect()
    torch.cuda.empty_cache()

    dataset = load_dataset("dim/open_orca_4475_DeepSeek-R1-Distill-Qwen-1.5B")
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=1000, seed=42)

    # test pass
    tokenize_single_turn(
        question=dataset["train"][0]["question"],
        answer=dataset["train"][0]["answer"],
        tokenizer=tokenizer,
    )
    train_examples = [
        tokenize_single_turn(tokenizer=tokenizer, **item)
        for item in tqdm(dataset["train"].to_list())
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

    dataset = Dataset.from_list(prepared_train_examples)
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
        skip_ids = [TEXT_TOKEN_ID, EOS_TOKEN_ID]
        for skip_id in skip_ids:
            padded_batch["labels"][padded_batch["labels"] == skip_id] = -100
        # print(padded_batch)
        return padded_batch

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=find_all_linear_names_v2(model=model),
    )

    formatted_date = datetime.fromtimestamp(time.time()).strftime(
        "%Y_%m_%d_%H_%M_%S_%f"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            num_train_epochs=2,  # Set this for 1 full training run.
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
            report_to="wandb",
            # report_to="none",
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            gradient_checkpointing=True,
            save_steps=10000,
            run_name=formatted_date,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
