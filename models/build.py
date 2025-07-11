import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, EulerDiscreteScheduler
from transformers import CLIPTokenizer, CLIPImageProcessor
from .modeling_clip import CustomCLIPTextModel
from .Florence2large.processing_florence2 import Florence2Processor
from .Florence2large.modeling_florence2 import Florence2ForConditionalGeneration
from .Florence2large.configuration_florence2 import Florence2Config

def load_sd_model(training_args):
    """Load Stable Diffusion model"""
    stable_diffusion_model_path = training_args.stable_diffusion_model_path
    text_encoder =  CustomCLIPTextModel.from_pretrained(stable_diffusion_model_path + "/text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_model_path + "/tokenizer")
    vae = AutoencoderKL.from_pretrained(stable_diffusion_model_path + "/vae",revision=None)
    scheduler = DDPMScheduler.from_pretrained(stable_diffusion_model_path + "/scheduler")
    unet = UNet2DConditionModel.from_pretrained(stable_diffusion_model_path + "/unet",revision=None)
    for m in [vae, text_encoder, unet]:
        for param in m.parameters():
            param.requires_grad = False

    return (vae, tokenizer, text_encoder, unet, scheduler)


def load_Florence2_model(training_args):
    florence2_model_path = training_args.florence2_model_path
    config = Florence2Config.from_pretrained(florence2_model_path)
    config.vision_config.model_type = "davit"
    config._attn_implementation = "eager"
    
    # Load the model
    model = Florence2ForConditionalGeneration.from_pretrained(
        florence2_model_path,
        config=config)
    
    processor = Florence2Processor.from_pretrained(
        florence2_model_path, config=config)

    # freeze the model
    if training_args.unfreeze_florence2_all:
        for param in model.parameters():
            param.requires_grad = True
    elif training_args.unfreeze_florence2_language_model:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.language_model.parameters():
            param.requires_grad = True
        for param in model.language_model.lm_head.parameters():
            param.requires_grad = False

        model.language_model.lm_head.weight = torch.nn.Parameter(
        model.language_model.lm_head.weight.detach().clone())

        for p in model.language_model.lm_head.parameters():
            p.requires_grad = False


    elif training_args.unfreeze_florence2_language_model_decoder:
        # Create a separate embedding layer for decoder
        original_embeddings = model.language_model.model.shared
        new_decoder_embeddings = torch.nn.Embedding(
            num_embeddings=original_embeddings.num_embeddings,
            embedding_dim=original_embeddings.embedding_dim,
            padding_idx=original_embeddings.padding_idx
        )
        # Copy the weights
        new_decoder_embeddings.weight.data = original_embeddings.weight.data.clone()
        
        # Replace the decoder embeddings
        model.language_model.model.encoder.embed_tokens = original_embeddings
        model.language_model.model.decoder.embed_tokens = new_decoder_embeddings
        for param in model.parameters():
            param.requires_grad = False
        for param in model.language_model.model.decoder.parameters():
            param.requires_grad = True
        model.language_model.model.decoder.embed_tokens.weight.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False

    return model, processor
