from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from .build import load_sd_model, load_Florence2_model
from .utils import initiate_time_steps, normalize
import torchvision.transforms as transforms
import os
from safetensors.torch import load_file
from .VLV_stage1 import SDModel
import argparse

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        
    def forward(self, x):
        return self.layers(x)


@dataclass
class CLIPDecoderOutput(ModelOutput):
    """
    Output class for the CLIP Decoder model.
    """
    last_hidden_state: Optional[torch.FloatTensor] = None
    generated_ids: Optional[torch.LongTensor] = None
    generated_text: Optional[list] = None


class CLIPDecoder(nn.Module):


    def __init__(
        self, 
        language_model: str,
        VLV_model: SDModel,
        device: torch.device,
        bf16: str,
        args: argparse.Namespace = None
    ):
        """
        Initialize the CLIP Decoder model.
        
        Args:
            language_model: Path to the language model
            VLV_model: The VLV model instance
            device: The device to run the model on
            bf16: Whether to use bfloat16 precision
        """
        super(CLIPDecoder, self).__init__()

        self._dtype = torch.bfloat16 if bf16 =="bf16" else torch.float32
        self.qwen2_tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.qwen2_model = AutoModelForCausalLM.from_pretrained(language_model,torch_dtype=self._dtype,device_map=None,low_cpu_mem_usage=True)
        self.VLV_model = VLV_model # fp32 in this case
        self._device = device
        self.mlp = MLP(input_dim=1024, output_dim=self.qwen2_model.config.hidden_size)
        self.ignore_token_id = -100

        self.qwen2_model = self.qwen2_model.to(self._device).to(self._dtype)
        self.VLV_model = self.VLV_model.to(self._device).to(self._dtype)
        self.mlp = self.mlp.to(self._device).to(self._dtype)
        self.qwen2_model.gradient_checkpointing_enable()

    
    
    def get_conditional_context(self, images, batch_size):
        """
        Get conditional context from images using the diffusion model.
        
        Args:
            images: Input images
            batch_size: Batch size
            
        Returns:
            Decoder hidden states from the diffusion model
        """
        prompt = ["<MORE_DETAILED_CAPTION>"] * batch_size
        inputs = self.VLV_model.processor(text=prompt, images=images, return_tensors="pt").to(self._device).to(self._dtype)
        
        if inputs["input_ids"] is not None:
            inputs_embeds = self.VLV_model.model.language_model.get_input_embeddings()(inputs["input_ids"]).to(self._device)
        
        if inputs["pixel_values"] is not None:
            image_features = self.VLV_model.model._encode_image(inputs["pixel_values"]).to(self._device)
            inputs_embeds, attention_mask = self.VLV_model.model._merge_input_ids_with_image_features(
                image_features, inputs_embeds
            )
        
        if inputs_embeds is not None:
            attention_mask = attention_mask.to(inputs_embeds.dtype)
            
        encoder_outputs = self.VLV_model.model.language_model.model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        decoder_inputs_embeds = self.VLV_model.query_embed.expand(batch_size, -1, -1)
        decoder_attention_mask = torch.ones(
            (batch_size, self.VLV_model.num_queries),
            dtype=self._dtype,
            device=self._device
        )

        encoder_hidden_states = encoder_outputs.last_hidden_state.to(self._dtype)
        decoder_input_embeds = decoder_inputs_embeds.to(self._dtype)
        attention_mask = attention_mask.to(self._dtype)

        decoder_outputs = self.VLV_model.model.language_model.model.decoder(
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
            
        return decoder_outputs.last_hidden_state
    
    def process_image(self, images, batch_size):
        """
        Process images to get clip text embeddings.
        
        Args:
            images: Input images
            batch_size: Batch size
            
        Returns:
            Processed clip text embeddings and attention mask
        """
        decoder_hidden_states = self.get_conditional_context(images, batch_size)
        context_embeds = self.VLV_model.language_proj(decoder_hidden_states).to(self._dtype)
        clip_text_embeds = self.VLV_model.text_encoder(inputs_embeds=context_embeds).last_hidden_state
        clip_text_embeds_output = self.mlp(clip_text_embeds)
        clip_text_embeds_attention_mask = torch.ones(
            (batch_size, self.VLV_model.num_queries),
            dtype=torch.long,
            device=self._device
        )
        
        return clip_text_embeds_output, clip_text_embeds_attention_mask
    
    def prepare_generation_inputs(self, clip_text_embeds, clip_text_attention_mask=None):
        """
        Prepare inputs for text generation.
        
        Args:
            clip_text_embeds: Processed clip text embeddings
            clip_text_attention_mask: Attention mask for clip text embeddings
            
        Returns:
            Dictionary of generation inputs
        """
        if clip_text_attention_mask is None:
            clip_text_attention_mask = torch.ones(
                (clip_text_embeds.shape[0], clip_text_embeds.shape[1]),
                dtype=torch.long,
                device=clip_text_embeds.device
            )
            
        return {
            "inputs_embeds": clip_text_embeds,
            "attention_mask": clip_text_attention_mask
        }
    
    def generate(self, images, max_new_tokens=300, num_beams=4, early_stopping=True):
        """
        Generate text from images.
        
        Args:
            images: Input images
            max_new_tokens: Maximum number of tokens to generate
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early in beam search
            
        Returns:
            CLIPDecoderOutput with generated ids and text
        """
        batch_size = len(images)
        clip_text_embeds, clip_text_attention_mask = self.process_image(images, batch_size)
        generation_inputs = self.prepare_generation_inputs(clip_text_embeds, clip_text_attention_mask)

        generation_inputs["inputs_embeds"] = generation_inputs["inputs_embeds"].to(torch.bfloat16)
        generation_inputs["attention_mask"] = generation_inputs["attention_mask"].to(torch.bfloat16)
    
        generated_ids = self.qwen2_model.generate(
            inputs_embeds=generation_inputs["inputs_embeds"],
            attention_mask=generation_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping
        )
        
        generated_text = self.qwen2_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return CLIPDecoderOutput(
            generated_ids=generated_ids,
            generated_text=generated_text
        )
    
    def forward(self, images, captions=None):
        """
        Forward pass for training.
        
        Args:
            images: Input images
            captions: Target captions (optional, for training)
            
        Returns:
            CLIPDecoderOutput with loss and logits
        """
        batch_size = images.shape[0]
        
        clip_text_embeds, clip_text_attention_mask = self.process_image(images, batch_size)
        
        if captions is None:
            return CLIPDecoderOutput(
                last_hidden_state=clip_text_embeds
            )
        assert len(captions) == batch_size
        
        qwen_input_ids = self.qwen2_tokenizer(
            text=captions,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
            max_length=300,
            return_token_type_ids=False,
        ).input_ids
        assert len(captions) == batch_size
        qwen_attention_mask = qwen_input_ids.ne(self.qwen2_tokenizer.pad_token_id).to(torch.long).to(self._device)
        
        labels = qwen_input_ids
        labels[labels == self.qwen2_tokenizer.pad_token_id] = self.ignore_token_id
        labels = labels.to(self._device)

        labels_for_embeddings = labels.clone()
        labels_for_embeddings[labels_for_embeddings == self.ignore_token_id] = self.qwen2_tokenizer.pad_token_id
        clip_text_embeds_qwen = self.qwen2_model.get_input_embeddings()(labels_for_embeddings)
        
        inputs_embeds = torch.cat((clip_text_embeds, clip_text_embeds_qwen), dim=1)
        clip_seq_len = clip_text_embeds.shape[1]
        clip_ignore_labels = torch.full((labels.shape[0], clip_seq_len), self.ignore_token_id).to(labels)
        combined_labels = torch.cat((clip_ignore_labels, labels), dim=1)
        
        attention_mask = torch.cat((
            clip_text_attention_mask,
            qwen_attention_mask
        ), dim=1)
        
        outputs = self.qwen2_model(
            inputs_embeds=inputs_embeds,
            labels=combined_labels,
            attention_mask=attention_mask,
            use_cache=False
        )
        return outputs