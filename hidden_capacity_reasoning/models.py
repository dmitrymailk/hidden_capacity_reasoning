from transformers import (
    Qwen2ForCausalLM,
    Qwen2Model,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
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
)


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


class Qwen2ModelEmbedPoolerV2(Qwen2ModelEmbedPoolerV1):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.lm_head = None
        self.post_init()


class Qwen2ModelEmbedPoolerV3(Qwen2ModelEmbedPoolerV1):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.lm_head = None
        self.weight_pooler = torch.nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.post_init()

    def forward(self, input_embeds):
        input_embeds = self.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )[0]
        # input_embeds = residual + input_embeds
        input_embeds = input_embeds.sum(1) / torch.tensor(
            input_embeds.shape[1],
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )
        # print(input_embeds.dtype)
        input_embeds = input_embeds.unsqueeze(1)
        # residual = input_embeds
        input_embeds = input_embeds.reshape(input_embeds.shape[0], 1, -1)
        input_embeds = self.weight_pooler(input_embeds)
        # input_embeds += residual
        # input_embeds = input_embeds.unsqueeze(1)
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


class Qwen2ForCausalLMCompressionV2(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.embed_pooler = Qwen2ModelEmbedPoolerV2(
            config=config,
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
            original_embeds = self.model.get_input_embeddings()(
                original_tokens_torch
            ).detach()
            # получаем эмбединги конечной формы которую мы будем моделировать
            compressed_embeds_template = self.model.get_input_embeddings()(
                compressed_tokens_torch
            ).detach()
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


class Qwen2ForCausalLMCompressionV3(Qwen2ForCausalLMCompressionV2):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.embed_pooler = Qwen2ModelEmbedPoolerV3(
            config=config,
        )

        # Initialize weights and apply final processing
        self.post_init()
