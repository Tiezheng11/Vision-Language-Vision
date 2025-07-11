import os
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from .build import load_sd_model, load_Florence2_model
from .utils import initiate_time_steps, normalize


class SDConfig(PretrainedConfig):
    """Configuration class for SDModel."""
    model_type = "sd"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        
    def forward(self, x):
        return self.layers(x)

@dataclass
class SDOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None

class SDModel(PreTrainedModel):
    config_class = SDConfig
    
    def __init__(
        self,
        config=None,
        training_args = None,
    ):
        if config is None:
            config = SDConfig()
        super().__init__(config)
        self.training_args = training_args
        if self.training_args.fp32:
            self._dtype = torch.float32
        else:
            self._dtype = torch.bfloat16
        self._device = torch.device(self.training_args.device if hasattr(self.training_args, 'device') else "cuda" if torch.cuda.is_available() else "cpu")
        
        self.vae, self.tokenizer, self.text_encoder, self.unet, self.scheduler = load_sd_model(training_args)
        torch.cuda.empty_cache()
        self.unet.eval()
        self.text_encoder.eval()
        self.model, self.processor = load_Florence2_model(training_args)

        self.unet = self.unet.to(self._dtype).to(self._device)
        self.text_encoder = self.text_encoder.to(self._dtype).to(self._device)
        self.model = self.model.to(self._dtype).to(self._device)
        self.vae = self.vae.to(torch.float32).to(self._device)

        self.batch_size = self.training_args.batch_size 

        hidden_dim = 1024 
        self.language_proj = nn.Sequential(
            nn.Linear(1024, hidden_dim, dtype=self._dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, 1024, dtype=self._dtype)
        ).to(self._device)
        for param in self.language_proj.parameters():
            param.requires_grad = True

        self.num_queries = self.training_args.learnable_token_length
        self.query_embed = nn.Parameter(torch.randn(1, self.num_queries, 1024, dtype=self._dtype))
        self.query_embed.requires_grad = True
        
        self.unet.enable_gradient_checkpointing()

    def _unet_pred_noise(self, x_start, t, noise, context):
        t = t.to(dtype=torch.long)
        
        dtype = self.unet.dtype
        x_start = x_start.to(dtype)
        noise = noise.to(dtype)
        context = context.to(dtype)
        
        nt = t.shape[0]
        noised_latent = self.scheduler.add_noise(x_start, noise, t)
        
        pred_noise = self.unet(
            noised_latent, 
            t, 
            encoder_hidden_states=context.expand(nt, -1, -1)
        ).sample

        return pred_noise
    
    def generate_images(self, images):
        batch_size = self.training_args.eval_batch_size
        prompt = ["<MORE_DETAILED_CAPTION>"] * batch_size
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self._device).to(self._dtype)
        
        if inputs["input_ids"] is not None:
            inputs_embeds = self.model.language_model.get_input_embeddings()(inputs["input_ids"]).to(self._dtype)
        if inputs["pixel_values"] is not None:
            image_features = self.model._encode_image(inputs["pixel_values"]).to(self._dtype)
            inputs_embeds, attention_mask = self.model._merge_input_ids_with_image_features(image_features, inputs_embeds)
        if inputs_embeds is not None:
            attention_mask = attention_mask.to(inputs_embeds.dtype)
        encoder_outputs = self.model.language_model.model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        decoder_input_embeds = self.query_embed.expand(batch_size, -1, -1)
        decoder_attention_mask = torch.ones(
            (batch_size, self.num_queries), 
            dtype=self._dtype, 
            device=self._device
        )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state.to(self._dtype)
        decoder_input_embeds = decoder_input_embeds.to(self._dtype)
        attention_mask = attention_mask.to(self._dtype)
        
        decoder_outputs = self.model.language_model.model.decoder(
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_decoder_hidden_state = decoder_outputs.last_hidden_state
        conditional_context = self.language_proj(last_decoder_hidden_state)

        un_token = self.tokenizer("", padding="max_length", truncation=True,max_length=77, return_tensors="pt").input_ids.to(self._device)
        un_context_embeddings = self.text_encoder(un_token).last_hidden_state
        un_context_embeddings = un_context_embeddings.expand(batch_size, -1, -1)
        if self.training_args.use_text_encoder:
            context_embeddings = self.text_encoder(
                inputs_embeds=conditional_context.to(self._dtype)
            ).last_hidden_state

        latent_shape = (batch_size, 4, self.training_args.image_size // 8, self.training_args.image_size // 8)
        latents = torch.randn(latent_shape, device=self._device, dtype=self._dtype)

        scheduler = self.scheduler
        scheduler.set_timesteps(self.training_args.num_inference_steps)
        with torch.no_grad():
            for t in scheduler.timesteps:
                latent_model_input = torch.cat([latents, latents], dim=0)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                combined_embeddings = torch.cat([un_context_embeddings, context_embeddings], dim=0).to(self._dtype)
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=combined_embeddings
                )[0]

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + self.training_args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents)[0]

        scaled_latents = latents / 0.18215
        with torch.no_grad():
            decoded_latents = self.vae.decode(scaled_latents.to(torch.float32))[0]

        return decoded_latents
    
    def get_conditional_context(self, images, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        prompt = ["<MORE_DETAILED_CAPTION>"] * batch_size
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self._device).to(self._dtype)
        
        if inputs["input_ids"] is not None:
            inputs_embeds = self.model.language_model.get_input_embeddings()(inputs["input_ids"]).to(self._dtype)
        if inputs["pixel_values"] is not None:
            image_features = self.model._encode_image(inputs["pixel_values"]).to(self._dtype)
            inputs_embeds, attention_mask = self.model._merge_input_ids_with_image_features(image_features, inputs_embeds)
        if inputs_embeds is not None:
            attention_mask = attention_mask.to(inputs_embeds.dtype)
        encoder_outputs = self.model.language_model.model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        decoder_input_embeds = self.query_embed.expand(batch_size, -1, -1)
        decoder_attention_mask = torch.ones(
            (batch_size, self.num_queries), 
            dtype=self._dtype, 
            device=self._device
        )
        
        encoder_hidden_states = encoder_outputs.last_hidden_state.to(self._dtype)
        decoder_input_embeds = decoder_input_embeds.to(self._dtype)
        attention_mask = attention_mask.to(self._dtype)
        
        decoder_outputs = self.model.language_model.model.decoder(
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_decoder_hidden_state = decoder_outputs.last_hidden_state
        return last_decoder_hidden_state
    
    def forward(
        self,
        image=None,
        filename=None,
        **kwargs,
    ) -> SDOutput:
        images_for_language_model = image
        normalize_images = normalize(image, rescale=True)
        x0=self.vae.encode(normalize_images.to(torch.float32)).latent_dist.sample()
        latent = x0 * 0.18215
        
        total_timestep = self.scheduler.num_train_timesteps

        timesteps = initiate_time_steps(0, total_timestep, self.batch_size, self.training_args).long()
        timesteps = timesteps.to(self._device)
        c, h, w = latent.shape[1:]
        if not self.training_args.use_same_noise_among_timesteps:
            noise = torch.randn((self.batch_size, c, h, w), device=self._device, dtype=self._dtype)
        else:
            noise = torch.randn((1, c, h, w), device=self._device, dtype=self._dtype)
            noise = noise.repeat(self.batch_size, 1, 1, 1)

        conditional_context = self.get_conditional_context(images_for_language_model)
        conditional_context = self.language_proj(conditional_context)

        if self.training_args.use_text_encoder:
            text_encoder_output = self.text_encoder(input_ids=None, inputs_embeds=conditional_context.to(self._dtype))
            pred_noise = self._unet_pred_noise(x_start=latent, t=timesteps, noise=noise, context=text_encoder_output.last_hidden_state.to(self._dtype)).to(self._dtype)
        else:
            pred_noise = self._unet_pred_noise(x_start=latent, t=timesteps, noise=noise, context=conditional_context.to(self._dtype)).to(self._dtype)
        
        if self.training_args.loss == "l1":
            loss = torch.nn.functional.l1_loss(pred_noise, noise)
        else:
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            
        return SDOutput(loss=loss)