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
<｜begin▁of▁sentence｜><｜User｜>ВОПРОС ОТ ПОЛЬЗОВАТЕЛЯ<｜Assistant｜><think>\n<|vision_start|><|fim_pad|><|vision_end|> МОДЕЛИ</think><|vision_start|><|fim_pad|><|vision_end|> КОНЕЧНОМУ ПОЛЬЗОВАТЕЛЮ
```
- где <|vision_start|><|fim_pad|><|vision_end|> это сжатая часть, где сжатие происходит с окнов в 10 токенов

