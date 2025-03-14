#### 2025_03_13_17_26_24_256272
- https://wandb.ai/dimweb/hidden_capacity_reasoning/runs/7ctsolll/overview
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
- Лора адаптер на 16, выкинуты ембединг слой и конечный линейный. Пулинг перевзвешенных эмбедингов основной модели. Основная модель полностью заморожена.

#### 2025_03_13_23_08_05_874474
- https://wandb.ai/dimweb/hidden_capacity_reasoning/runs/ayvk7z1w/overview
- тоже самое что и в 2025_03_13_17_26_24_256272, только
- train_batch_size 2
- accumulation 8