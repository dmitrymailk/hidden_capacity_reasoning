import numpy as np

TEXT_TOKEN_ID = 151662
EOS_TOKEN_ID = 151643
WINDOW_SIZE = 4


def tokenize_single_turn(
    question: str = None,
    answer: str = None,
    tokenizer=None,
):
    """
    tokenization for r1
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    """
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
    part_4 = answer
    content_compression_mask += len(
        tokenizer.encode(
            part_4,
            add_special_tokens=False,
        )
    ) * [2]

    part_5 = "<｜end▁of▁sentence｜>"
    content_compression_mask += len(
        tokenizer.encode(
            part_5,
            add_special_tokens=False,
        )
    ) * [0]

    complete_prompt = ""
    for part in [part_1, part_2, part_3, part_4, part_5]:
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
    tokenizer=None,
    text_token_id: int = TEXT_TOKEN_ID,
    window_size: int = WINDOW_SIZE,
    train_examples_amount: int = -1,
):
    aligned_batch = []
    # определяем какие именно элементы мы хотим сжимать,а какие
    # не помещаются не помещаются в чанк размером window_size
    # TODO: написать обработку крайних случаев
    for tokens in dataset_batch:
        input_ids = np.array(tokens["input_ids"])
        content_mask = np.array(tokens["content_compression_mask"])
        user_part = input_ids[content_mask == 1]
        total_parts = len(user_part) // window_size
        new_len_part_1 = total_parts * window_size
        mask_end_pos = np.where(content_mask == 1)[0][-1]
        # print(content_mask.tolist())
        content_mask[
            mask_end_pos - (len(user_part) - new_len_part_1) + 1 : mask_end_pos + 1
        ] = 0
        # print(content_mask.tolist())
        # print(user_part.shape, total_parts, new_len_part_1)

        answer_part = input_ids[content_mask == 2]
        total_parts = len(answer_part) // window_size
        new_len_part_2 = total_parts * window_size
        mask_end_pos = np.where(content_mask == 2)[0][-1]
        # print(content_mask.tolist())
        content_mask[
            mask_end_pos - (len(answer_part) - new_len_part_2) + 1 : mask_end_pos + 1
        ] = 0
        # print(content_mask.tolist())
        # print(answer_part.shape, total_parts, new_len_part_2)
        # content_mask[content_mask == 2] = 1
        # print(content_mask.tolist())
        aligned_batch.append(
            {
                "input_ids": tokens["input_ids"],
                "content_compression_mask": content_mask,
                "attention_mask": tokens["attention_mask"],
            }
        )
        # break

    train_examples = []
    for tokens in aligned_batch:
        input_ids = np.array(tokens["input_ids"])
        content_mask = np.array(tokens["content_compression_mask"])
        if train_examples_amount == -1:
            # -1 от всех кусков, потому что если мы сожмем все части, моделировать
            # будет нечего
            train_examples_amount = (
                content_mask[content_mask == 2].shape[0] // window_size - 1
            )
        for chunks_amount in range(train_examples_amount):
            # фикусируемся на сжатии ответа модели, выбираем 2
            start_pos = np.where(content_mask == 2)[0][0]
            input_ids[start_pos : start_pos + (chunks_amount + 1) * window_size] = (
                text_token_id
            )
            compressed_input_ids = input_ids[:start_pos].tolist()
            compressed_input_ids += [text_token_id] * (chunks_amount + 1)
            compressed_input_ids += input_ids[
                start_pos + (chunks_amount + 1) * window_size :
            ].tolist()
            # print(text_token_id)
            # print(" ".join(f"{num:>{8}}" for num in content_mask.tolist()))
            # print(" ".join(f"{num:>{8}}" for num in input_ids.tolist()))
            # print(" ".join(f"{num:>{8}}" for num in compressed_input_ids))
            # print("===")
            train_examples.append(
                {
                    "replaced_original_tokens": input_ids.tolist(),
                    "compressed_input_ids": compressed_input_ids,
                    "original_tokens": tokens["input_ids"],
                }
            )
        # break
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
