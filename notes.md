#### 2025_03_17_01_39_35_074194
- https://wandb.ai/dimweb/hidden_capacity_reasoning/runs/opw3q7yw?nw=nwuserdimweb
```python
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
```
- train_batch_size 1
- accumulation 4
- Лора адаптер на 16, выкинут конечный линейный. Пулинг перевзвешенных эмбедингов основной модели. Основная модель полностью заморожена.
- также тренируются эмбединги

```python
# получаем оригинальные токены
original_tokens_torch = kwargs["original_tokens"].to(self.model.device)
#  получаем токены с местами которые мы хотим сжать
replaced_tokens_torch = kwargs["replaced_original_tokens"].to(
    self.model.device
)
# получаем итоговую форму токенов которая должна получиться
compressed_tokens_torch = kwargs["compressed_input_ids"].to(
    self.model.device
)

# получаем оригинальные эмбединги из рассуждающей модели
original_embeds = self.model.get_input_embeddings()(original_tokens_torch)
# получаем эмбединги конечной формы которую мы будем моделировать
compressed_embeds_template = self.model.get_input_embeddings()(
    compressed_tokens_torch
)
# создаем маску из токенов которые были помечены на сжатие
tokens_for_compression_mask = replaced_tokens_torch == TEXT_TOKEN_ID
# получаем маску итоговых сжатых токенов куда нам нужно будет положить
# после пулинга
compressed_tokens_mask = compressed_tokens_torch == TEXT_TOKEN_ID
# выбираем оригинальные токены для компрессии
tokens_for_compression = original_tokens_torch[tokens_for_compression_mask]
# получаем эмбединги оригинальных токенов из модели для пулинга
# так как эти эмбединги мы обучаем
embeds_for_compression = self.embed_pooler.model.get_input_embeddings()(
    tokens_for_compression
)
# решейпим эмбединги для их пулинга
embeds_for_compression = embeds_for_compression.reshape(
    -1,
    WINDOW_SIZE,
    original_embeds.shape[-1],
)
# сжимаем эмбединги
pooled_embeds = self.embed_pooler(embeds_for_compression)
# кастуем
pooled_embeds = pooled_embeds.to(compressed_embeds_template.dtype)
# помещаем сжатые ембединги на соответствующие места, чтобы они соседствовали рядом
# с эмбедингами оригинальной(необучаемой) модели
compressed_embeds_template = compressed_embeds_template.masked_scatter(
    compressed_tokens_mask.unsqueeze(-1).expand_as(
        compressed_embeds_template
    ),
    pooled_embeds,
)
# создаем маску для ембедингов начала и конца
compressed_special_tokens_mask = (
    compressed_tokens_torch == VISION_START
) | (compressed_tokens_torch == VISION_END)
# выбираем нужные токены
compressed_special_tokens = compressed_tokens_torch[
    compressed_special_tokens_mask
]
# получаем их эмбединги
compressed_special_embeds = self.embed_pooler.model.get_input_embeddings()(
    compressed_special_tokens
).to(compressed_embeds_template.dtype)
# копируем их в нужное место
compressed_embeds_template = compressed_embeds_template.masked_scatter_(
    compressed_special_tokens_mask.unsqueeze(-1).expand_as(
        compressed_embeds_template
    ),
    compressed_special_embeds,
)
inputs_embeds = compressed_embeds_template
```

- изначальная входная последовательность выглядит так
```text
<｜begin▁of▁sentence｜><｜User｜>ВОПРОС ОТ ПОЛЬЗОВАТЕЛЯ<｜Assistant｜><think>\nРАЗМЫШЛЕНИЯ МОДЕЛИ</think>ОТВЕТ КОНЕЧНОМУ ПОЛЬЗОВАТЕЛЮ
```
- я же ее переделал в
```text
<｜begin▁of▁sentence｜><｜User｜>ВОПРОС ОТ ПОЛЬЗОВАТЕЛЯ<｜Assistant｜><think>\n<|vision_start|><|fim_pad|><|vision_end|> МОДЕЛИ</think>ОТВЕТ КОНЕЧНОМУ ПОЛЬЗОВАТЕЛЮ
```
```text
<｜begin▁of▁sentence｜><｜User｜>ВОПРОС ОТ ПОЛЬЗОВАТЕЛЯ<｜Assistant｜><think>\n<|vision_start|><|fim_pad|><|fim_pad|><|vision_end|></think><|vision_start|><|fim_pad|><|vision_end|> КОНЕЧНОМУ ПОЛЬЗОВАТЕЛЮ
```
```text
<｜begin▁of▁sentence｜><｜User｜>ВОПРОС ОТ ПОЛЬЗОВАТЕЛЯ<｜Assistant｜><think>\n<|vision_start|><|fim_pad|><|fim_pad|><|vision_end|></think><|vision_start|><|fim_pad|><|fim_pad|><|vision_end|> ПОЛЬЗОВАТЕЛЮ
```
- где <|vision_start|><|fim_pad|><|vision_end|> это сжатая часть, где сжатие происходит с окнов в 10 токенов

##### Инференс 
- выглядит следующим образом. генерим токенов размер окна+еще доп несколько токенов
```python
from hidden_capacity_reasoning.utils import WINDOW_SIZE, VISION_START, VISION_END


# model = trainer.model
generated_tokens = tokenizer.apply_chat_template(
    [
        # {"role": "user", "content": "how many wings has a bird?"},
        {"role": "user", "content": dataset["train"].to_list()[:5][0]["question"]},
    ],
    tokenize=True,
    add_generation_prompt=True,
)

with torch.no_grad(), torch.autocast(device_type="cuda"):
    start_embed = model.base_model.embed_pooler.model.get_input_embeddings()(
        torch.tensor([[VISION_START]], device="cuda")
    )
    end_embed = model.base_model.embed_pooler.model.get_input_embeddings()(
        torch.tensor([[VISION_END]], device="cuda")
    )
    generated_tokens = torch.tensor(generated_tokens).unsqueeze(0).cuda()
    generated_embeds = model.get_input_embeddings()(generated_tokens)
    temp_gen_size = 0
    window_size = WINDOW_SIZE  # + 1
    new_tokens = 4
    generation_started = False
    max_steps = (new_tokens + window_size) * 5
    print("generated_embeds", generated_embeds.shape)
    for step in range(max_steps):
        if temp_gen_size == window_size + new_tokens:
            print(
                "TOKENS FOR EMDED",
                tokenizer.decode(
                    generated_tokens[:, -(window_size + new_tokens) :][:, :WINDOW_SIZE]
                    .cpu()
                    .tolist()[0]
                ),
            )
            # tokenizer.decode(generated_tokens[:, : -window_size ].cpu().tolist()[0])
            if hasattr(model.base_model, "embed_pooler"):
                new_embeds_for_compression = (
                    model.base_model.embed_pooler.model.get_input_embeddings()(
                        generated_tokens[:, -(window_size + new_tokens) :][
                            :, :WINDOW_SIZE
                        ]
                    )
                ).to(torch.bfloat16)
                compressed_part = model.base_model.embed_pooler(
                    new_embeds_for_compression
                )
            else:
                compressed_part = model.embed_pooler(new_embeds_for_compression)

            if generation_started:
                generated_embeds = torch.cat(
                    [
                        generated_embeds[:, : -(window_size + new_tokens + 1)],
                        compressed_part,
                        end_embed,
                        # torch.randn(1, 1, 1536, device="cuda"),
                        generated_embeds[:, -new_tokens:],
                    ],
                    dim=1,
                )
            else:
                generated_embeds = torch.cat(
                    [
                        generated_embeds[:, : -(window_size + new_tokens)],
                        start_embed,
                        compressed_part,
                        end_embed,
                        # torch.randn(1, 1, 1536, device="cuda"),
                        generated_embeds[:, -new_tokens:],
                    ],
                    dim=1,
                )
                generation_started = True
            temp_gen_size = 1

        logits = model(
            inputs_embeds=generated_embeds,
        ).logits
        top_token = logits.argmax(-1)[-1][-1]
        top_token_embed = model.get_input_embeddings()(top_token)
        # print(top)
        generated_tokens = torch.cat([generated_tokens, top_token.reshape(1, 1)], dim=1)

        generated_embeds = torch.cat(
            [generated_embeds, top_token_embed.reshape(1, 1, -1)], dim=1
        )
        print(temp_gen_size, tokenizer.decode(generated_tokens[-1]))

        temp_gen_size += 1

    # print(tokenizer.decode(generated_tokens[-1]))

# break
```
```text
Okay, so I need to figure out whether the statement "In 1863, Robert E. Lee's Confederate incursion north ended at the Battle of Gettysburg." is a reasonable answer to the question: "What date did the American Civil War start? I remember that the Civil War started in 1863.
```
- original text
```text
Okay, so I need to figure out whether the statement "In 1863, Robert E. Lee's Confederate incursion north ended at the Battle of Gettysburg." is a reasonable answer to the question: "What date did the American Civil War start?" 

First, I should recall when the American Civil War actually started. I remember that the Civil War began in 1861 when states seceded from the Union to form the Confederate states. So the start date is 1861.
```
##### sad story
- данная генерация это пример на обучающих данных, на которых обучалась модель. причем предложений для обучения в обучающей выборке было 3, а количество эпох 90, но даже так мы не можем полностью запомнить данные, когда обучаем только сжимающую модель.


### 2025_03_19_13_29_15_704211
- Сократил обучающую выборку до первых 2000 предложений
- сократил окно, до 2 токенов