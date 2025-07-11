from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPTextModel, CLIPPreTrainedModel, CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPTextEmbeddings, CLIPEncoder, CLIPAttention, CLIPMLP, CLIPEncoderLayer, _create_4d_causal_attention_mask, _prepare_4d_attention_mask, BaseModelOutputWithPooling
from typing import Optional, Union, Tuple
import torch
from torch import nn

class CustomCLIPTokenizer(CLIPTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inherit everything from the original tokenizer
        # No additional initialization needed unless you want to add specific features

class CustomCLIPImageProcessor(CLIPImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inherit everything from the original processor
        # No additional initialization needed unless you want to add specific features

class CustomCLIPTextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either input_ids or inputs_embeds")


        if inputs_embeds is not None:
            inputs_embeds = self.embeddings(inputs_embeds=inputs_embeds)
        else:
            inputs_embeds = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        causal_attention_mask = _create_4d_causal_attention_mask(
            inputs_embeds.size()[:-1], inputs_embeds.dtype, device=inputs_embeds.device
        )
        
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Update the pooled output computation to work with both input types
        if input_ids is not None:
            # Use input_ids to find the EOS token position
            if self.eos_token_id == 2:
                pooled_output = last_hidden_state[
                    torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                    input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                ]
            else:
                pooled_output = last_hidden_state[
                    torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                    (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                    .int()
                    .argmax(dim=-1),
                ]
        else:
            # When using inputs_embeds, use the last token as the pooled output
            pooled_output = last_hidden_state[:, -1]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class CustomCLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CustomCLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
