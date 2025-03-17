import numpy as np
import torch
from more_itertools import chunked, collapse

TEXT_TOKEN_ID = 151662
EOS_TOKEN_ID = 151643
WINDOW_SIZE = 10
VISION_START = 151652
VISION_END = 151653
END_THINK_ID = 151649


def tokenize_single_turn(
    question,
    answer,
    tokenizer=None,
):
    content_compression_mask = []

    part_1 = """<｜begin▁of▁sentence｜><｜User｜>"""
    content_compression_mask += len(
        tokenizer.encode(
            part_1,
            add_special_tokens=False,
        )
    ) * [0]

    # question
    part_2 = question
    content_compression_mask += len(
        tokenizer.encode(
            part_2,
            add_special_tokens=False,
        )
    ) * [1]

    part_3 = "<｜Assistant｜><think>\n"
    content_compression_mask += len(
        tokenizer.encode(
            part_3,
            add_special_tokens=False,
        )
    ) * [0]

    # answer
    part_4, part_6 = answer.split("</think>")

    content_compression_mask += len(
        tokenizer.encode(
            part_4,
            add_special_tokens=False,
        )
    ) * [2]
    # </think>
    part_5 = "</think>"
    content_compression_mask += [0]
    content_compression_mask += len(
        tokenizer.encode(
            part_6,
            add_special_tokens=False,
        )
    ) * [3]

    part_7 = "<｜end▁of▁sentence｜>"
    content_compression_mask += [0]

    complete_prompt = ""
    for part in [
        part_1,
        part_2,
        part_3,
        part_4,
        part_5,
        part_6,
        part_7,
    ]:
        complete_prompt += part
    original_tokens = tokenizer.encode(
        complete_prompt,
        add_special_tokens=False,
    )
    attention_mask = len(original_tokens) * [1]
    return {
        "input_ids": original_tokens,
        "attention_mask": attention_mask,
        "content_compression_mask": content_compression_mask,
    }


def generate_train_examples(
    dataset_batch: list,
    text_token_id: int = TEXT_TOKEN_ID,
    window_size: int = WINDOW_SIZE,
    train_examples_amount: int = -1,
):
    """
    1 - часть вопроса
    2 - часть размышлений
    3 - часть ответа пользователю
    """
    train_examples = []
    # определяем какие именно элементы мы хотим сжимать,а какие
    # не помещаются не помещаются в чанк размером window_size
    # TODO: написать обработку крайних случаев
    for tokens in dataset_batch:
        input_ids = np.array(tokens["input_ids"])
        content_mask = np.array(tokens["content_compression_mask"])

        # рассуждение
        mask_pos_2 = np.where(content_mask == 2)[0]
        start_pos_2 = mask_pos_2[0]
        end_pos_2 = mask_pos_2[-1]
        # ответ
        mask_pos_3 = np.where(content_mask == 3)[0]
        start_pos_3 = mask_pos_3[0]
        end_pos_3 = mask_pos_3[-1]

        part_2_chunks_pad = []
        for part_2 in chunked(input_ids[mask_pos_2].tolist(), window_size):
            if len(part_2) % window_size != 0:
                part_2 += (window_size - len(part_2)) * [EOS_TOKEN_ID]
            part_2_chunks_pad.append(part_2)
        part_2_chunks = list(chunked(input_ids[mask_pos_2].tolist(), window_size))

        part_3_chunks = []
        # сжимаем все части кроме последнего чанка
        part_3_chunks = list(chunked(input_ids[mask_pos_3].tolist(), window_size))
        total_chunks = part_2_chunks + part_3_chunks
        for compress_parts_amount in range(1, len(total_chunks) - 1):
            if compress_parts_amount <= len(part_2_chunks):
                if compress_parts_amount == len(part_2_chunks):
                    part_2_chunks = part_2_chunks_pad
                compress_tokens = [text_token_id] * compress_parts_amount
                compress_tokens = [VISION_START] + compress_tokens + [VISION_END]
                compressed_input_ids = input_ids[:start_pos_2].tolist()
                compressed_input_ids += compress_tokens

                replaced_original_tokens = input_ids[:start_pos_2].tolist()
                replaced_original_tokens += (
                    [VISION_START]
                    + compress_parts_amount * [text_token_id] * window_size
                    + [VISION_END]
                )

                new_original_tokens = input_ids[:start_pos_2].tolist()
                new_original_tokens += (
                    [VISION_START]
                    + list(collapse(part_2_chunks[:compress_parts_amount]))
                    + [VISION_END]
                )

                remain_tokens_2 = list(collapse(part_2_chunks[compress_parts_amount:]))
                compressed_input_ids += remain_tokens_2
                compressed_input_ids += [END_THINK_ID]

                replaced_original_tokens += remain_tokens_2
                replaced_original_tokens += [END_THINK_ID]

                new_original_tokens += remain_tokens_2
                new_original_tokens += [END_THINK_ID]

                remain_tokens_3 = list(collapse(part_3_chunks))
                compressed_input_ids += remain_tokens_3
                compressed_input_ids += [EOS_TOKEN_ID]

                replaced_original_tokens += remain_tokens_3
                replaced_original_tokens += [EOS_TOKEN_ID]

                new_original_tokens += remain_tokens_3
                new_original_tokens += [EOS_TOKEN_ID]

                train_examples.append(
                    {
                        "replaced_original_tokens": replaced_original_tokens,
                        "compressed_input_ids": compressed_input_ids,
                        "original_tokens": new_original_tokens,
                    }
                )
            # но зачем сжимать часть с ответом, она всегда кратно меньше
            else:
                # print(tokenizer.decode(replaced_original_tokens))
                # break
                # compressed_input_ids
                # сжимаем всю рассуждающую часть
                compressed_input_ids = input_ids[:start_pos_2].tolist()
                compress_tokens = [text_token_id] * len(part_2_chunks_pad)
                compress_tokens = [VISION_START] + compress_tokens + [VISION_END]
                compressed_input_ids += compress_tokens
                compressed_input_ids += [END_THINK_ID]

                # replaced_original_tokens
                replaced_original_tokens = input_ids[:start_pos_2].tolist()
                replaced_original_tokens += (
                    [VISION_START]
                    + [text_token_id] * len(part_2_chunks_pad) * window_size
                    + [VISION_END]
                )
                replaced_original_tokens += [END_THINK_ID]

                # new_original_tokens
                new_original_tokens = input_ids[:start_pos_2].tolist()
                new_original_tokens += (
                    [VISION_START] + list(collapse(part_2_chunks_pad)) + [VISION_END]
                )
                new_original_tokens += [END_THINK_ID]

                # compressed_input_ids
                compress_tokens = [text_token_id] * (
                    compress_parts_amount - len(part_2_chunks_pad)
                )
                compress_tokens = [VISION_START] + compress_tokens + [VISION_END]
                compressed_input_ids += compress_tokens
                compressed_input_ids += list(
                    collapse(
                        part_3_chunks[compress_parts_amount - len(part_2_chunks_pad) :]
                    )
                )
                compressed_input_ids += [EOS_TOKEN_ID]

                # replaced_original_tokens
                compress_tokens = (
                    [text_token_id]
                    * (compress_parts_amount - len(part_2_chunks_pad))
                    * window_size
                )
                replaced_original_tokens += (
                    [VISION_START] + compress_tokens + [VISION_END]
                )
                replaced_original_tokens += list(
                    collapse(
                        part_3_chunks[compress_parts_amount - len(part_2_chunks_pad) :]
                    )
                )
                replaced_original_tokens += [EOS_TOKEN_ID]

                # new_original_tokens
                compress_tokens = list(
                    collapse(
                        part_3_chunks[: compress_parts_amount - len(part_2_chunks_pad)]
                    )
                )
                new_original_tokens += [VISION_START] + compress_tokens + [VISION_END]
                new_original_tokens += list(
                    collapse(
                        part_3_chunks[compress_parts_amount - len(part_2_chunks_pad) :]
                    )
                )
                new_original_tokens += [EOS_TOKEN_ID]

                train_examples.append(
                    {
                        "replaced_original_tokens": replaced_original_tokens,
                        "compressed_input_ids": compressed_input_ids,
                        "original_tokens": new_original_tokens,
                    }
                )
    return train_examples


def pad_train_examples(
    train_examples: list,
    tokenizer: None,
):
    # pad to the same length
    new_inputs = {}
    for item in train_examples:
        for key, value in item.items():
            if not key in new_inputs:
                new_inputs[key] = []
            new_inputs[key].append(value)

    for key, value in new_inputs.items():
        new_inputs[key] = tokenizer.pad(
            {
                "input_ids": new_inputs[key],
            },
            padding=True,
            # return_tensors="pt",
        )
    return new_inputs


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


def find_all_linear_names_v3(model):
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
            "embed_tokens",
        ]
    )
    for name, module in model.named_modules():
        if "embed_pooler" in name:
            names = name.split(".")[-1]
            if names in target_modules:
                lora_module_names.add(name)
        # if isinstance(module, torch.nn.Linear):
    return lora_module_names
