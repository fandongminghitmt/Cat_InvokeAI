from typing import Literal, Optional, List, Union, Dict, Any
import torch
import os
from PIL import Image
import numpy as np

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InvocationContext,
    invocation,
    invocation_output,
    BaseInvocationOutput
)
from invokeai.app.invocations.fields import (
    ImageField,
    InputField,
    WithMetadata,
    WithBoard,
    FieldDescriptions,
    OutputField,
    ConditioningField,
    UIComponent
)
from invokeai.backend.util.devices import choose_torch_device, torch_dtype

# Global Cache to simulate passing model objects between nodes
# In a proper integration, this would use InvokeAI's ModelManager
_VNCCS_CACHE = {
    "unet": {},
    "clip": {},
    "vae": {},
    "lora": {},
    "conditioning": {},
    "latents": {}
}

def get_cached_model(cache_type: str, key: str):
    return _VNCCS_CACHE.get(cache_type, {}).get(key)

def set_cached_model(cache_type: str, key: str, model: Any):
    if cache_type not in _VNCCS_CACHE:
        _VNCCS_CACHE[cache_type] = {}
    _VNCCS_CACHE[cache_type][key] = model

def wrap_model(model_obj: Any) -> Dict[str, Any]:
    """Ensures model is wrapped in a dict to support config patching"""
    if isinstance(model_obj, dict) and "model" in model_obj:
        return model_obj
    return {"model": model_obj, "config": {}, "patches": []}

# --- Output Types ---

@invocation_output("vnccs_model_output")
class VNCCSModelOutput(BaseInvocationOutput):
    """Output for Model Loaders"""
    model_key: str = OutputField(description="Key to retrieve the loaded model from cache")

@invocation_output("vnccs_vae_output")
class VNCCSVAEOutput(BaseInvocationOutput):
    """Output for VAE Loader"""
    vae_key: str = OutputField(description="Key to retrieve the loaded VAE")

@invocation_output("vnccs_clip_output")
class VNCCSCLIPOutput(BaseInvocationOutput):
    """Output for CLIP/TextEncoder Loader"""
    clip_key: str = OutputField(description="Key to retrieve the loaded CLIP/Qwen model")

@invocation_output("vnccs_conditioning_output")
class VNCCSConditioningOutput(BaseInvocationOutput):
    """Output for Encoder"""
    positive_conditioning: str = OutputField(description="Positive Conditioning (Key/Object)")
    negative_conditioning: str = OutputField(description="Negative Conditioning (Key/Object)")

@invocation_output("vnccs_latent_output")
class VNCCSLatentOutput(BaseInvocationOutput):
    """Output for Sampler"""
    latents: str = OutputField(description="Latents (Key/Object)")

@invocation_output("vnccs_image_output")
class VNCCSImageOutput(BaseInvocationOutput):
    """Output for Image"""
    image: ImageField = OutputField(description="The output image")

# Helper for SDXL models
SDXL_DIR = r"D:\Cat_InvokeAI\SDXL"

def get_sdxl_models():
    if not os.path.exists(SDXL_DIR):
        return ["None"]
    models = []
    try:
        for name in os.listdir(SDXL_DIR):
            models.append(name)
    except Exception:
        pass
    return sorted(models) if models else ["None"]

SDXL_MODELS = get_sdxl_models()
# Create Literal dynamically
SDXLModelsLiteral = Literal.__getitem__(tuple(SDXL_MODELS))

# --- 1. UNet Loader ---
@invocation("vnccs_unet_loader", title="VNCCS UNet Loader", tags=["vnccs", "unet", "loader"], category="vnccs", version="1.0.2")
class VNCCSUNetLoaderInvocation(BaseInvocation):
    """Loads a UNet model (AuraFlow or other) using ComfyUI Logic"""
    model_path: SDXLModelsLiteral = InputField(default=SDXL_MODELS[0] if SDXL_MODELS else "None", description="Select Model from SDXL Directory")
    
    def invoke(self, context: InvocationContext) -> VNCCSModelOutput:
        import comfy.sd
        import comfy.utils
        
        full_path = os.path.join(SDXL_DIR, self.model_path)
        key = full_path
        
        if not get_cached_model("unet", key):
            print(f"Loading UNet from {full_path} using ComfyUI...")
            try:
                # ComfyUI Checkpoint Loader
                # load_checkpoint_guess_config returns (model, clip, vae, clipvision)
                # We only need the model (ModelPatcher)
                
                # Note: output_vae=False, output_clip=False avoids loading them if not needed
                out = comfy.sd.load_checkpoint_guess_config(full_path, output_vae=False, output_clip=False, embedding_directory=None)
                model_patcher = out[0] # This is the ModelPatcher object
                
                # Wrap it to support our custom pipeline's config patching mechanism
                # The downstream nodes expect {"model": obj, "config": {...}}
                wrapper = wrap_model(model_patcher)
                
                set_cached_model("unet", key, wrapper)
                
            except Exception as e:
                print(f"Failed to load UNet with ComfyUI: {e}")
                # Fallback logic for Diffusers directory if Comfy fails (unlikely if path is correct)
                # But to be safe, we can try the old way if it's a directory
                if os.path.isdir(full_path):
                     print("Trying fallback to diffusers for directory...")
                     from diffusers import Transformer2DModel, AuraFlowTransformer2DModel
                     if "aura" in full_path.lower():
                        model = AuraFlowTransformer2DModel.from_pretrained(full_path, torch_dtype=torch.bfloat16).to("cuda")
                     else:
                        model = Transformer2DModel.from_pretrained(full_path, torch_dtype=torch.float16).to("cuda")
                     set_cached_model("unet", key, wrap_model(model))
                else:
                     raise
                
        return VNCCSModelOutput(model_key=key)

# --- 2. LoRA Loader ---
@invocation("vnccs_lora_loader", title="VNCCS LoRA Loader (Model)", tags=["vnccs", "lora"], category="vnccs", version="1.0.0")
class VNCCSLoRALoaderInvocation(BaseInvocation):
    """Loads a LoRA and applies it to the UNet"""
    unet_key: str = InputField(description="The UNet model key")
    lora_path: str = InputField(description="Path to LoRA file")
    strength: float = InputField(default=1.0, description="LoRA strength")
    
    def invoke(self, context: InvocationContext) -> VNCCSModelOutput:
        unet_wrapper = get_cached_model("unet", self.unet_key)
        if not unet_wrapper:
            raise ValueError("UNet not found in cache")
            
        # Mock implementation: Real LoRA loading requires peft or manual weight injection
        print(f"Applying LoRA from {self.lora_path} with strength {self.strength} (Mock)")
        
        # In a real implementation, we would apply lora to a clone or adapter
        # Here we just pass the key through, assuming in-place or no-op for mock
        
        return VNCCSModelOutput(model_key=self.unet_key)

# --- 3. AuraFlow Sampler (Model Patch) ---
@invocation("vnccs_auraflow_sampler", title="VNCCS Sampler (AuraFlow)", tags=["vnccs", "sampler", "patch"], category="vnccs", version="1.0.0")
class VNCCSAuraFlowSamplerInvocation(BaseInvocation):
    """Patches the model with AuraFlow specific sampling parameters (Shift)"""
    model_key: str = InputField(description="Model Key to patch")
    shift: float = InputField(default=1.73, description="Flow Matching Shift parameter")
    
    def invoke(self, context: InvocationContext) -> VNCCSModelOutput:
        entry = get_cached_model("unet", self.model_key)
        if not entry:
            raise ValueError("Model not found in cache")
            
        # Ensure it's a wrapper
        entry = wrap_model(entry)
        
        # Create a shallow copy for the patch chain
        new_entry = entry.copy()
        new_entry["config"] = entry["config"].copy()
        
        # Apply patch
        new_entry["config"]["shift"] = self.shift
        print(f"Patched Model with AuraFlow Shift: {self.shift}")
        
        new_key = f"{self.model_key}_aura_{context.graph_execution_state_id}"
        set_cached_model("unet", new_key, new_entry)
        
        return VNCCSModelOutput(model_key=new_key)

# --- 4. CFG Normalization (Model Patch) ---
@invocation("vnccs_cfg_norm", title="VNCCS CFG Normalization", tags=["vnccs", "cfg", "patch"], category="vnccs", version="1.0.0")
class VNCCSCFGUpscaleInvocation(BaseInvocation):
    """Patches the model with CFG Normalization / Rescale CFG"""
    model_key: str = InputField(description="Model Key to patch")
    rescale_factor: float = InputField(default=0.7, description="Rescale factor")
    
    def invoke(self, context: InvocationContext) -> VNCCSModelOutput:
        entry = get_cached_model("unet", self.model_key)
        if not entry:
            raise ValueError("Model not found in cache")
            
        # Ensure it's a wrapper
        entry = wrap_model(entry)
        
        # Create a shallow copy for the patch chain
        new_entry = entry.copy()
        new_entry["config"] = entry["config"].copy()
        
        # Apply patch
        new_entry["config"]["cfg_rescale"] = self.rescale_factor
        print(f"Patched Model with CFG Rescale: {self.rescale_factor}")
        
        new_key = f"{self.model_key}_cfgnorm_{context.graph_execution_state_id}"
        set_cached_model("unet", new_key, new_entry)
        
        return VNCCSModelOutput(model_key=new_key)

# --- 5. Load CLIP (Qwen) ---


@invocation("vnccs_load_clip", title="VNCCS Load CLIP (Qwen)", tags=["vnccs", "clip", "qwen"], category="vnccs", version="1.0.3")
class VNCCSLoadCLIPInvocation(BaseInvocation):
    """Loads the Qwen model to act as CLIP/Text Encoder"""
    model_path: SDXLModelsLiteral = InputField(default=SDXL_MODELS[0] if SDXL_MODELS else "None", description="Select Model from SDXL Directory")
    
    def invoke(self, context: InvocationContext) -> VNCCSCLIPOutput:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLConfig
        import comfy.utils
        
        full_path = os.path.join(SDXL_DIR, self.model_path)
        key = full_path
        
        if not get_cached_model("clip", key):
            print(f"Loading Qwen-VL as CLIP from {full_path}...")
            
            try:
                if os.path.isfile(full_path):
                    print("Detected single file. Using ComfyUI file loader...")
                    
                    # 1. Load state dict using ComfyUI (handles safetensors, pth, device)
                    state_dict = comfy.utils.load_torch_file(full_path)
                    
                    # 2. Config
                    config_id = "Qwen/Qwen2.5-VL-7B-Instruct" 
                    try:
                        config = Qwen2_5_VLConfig.from_pretrained(config_id)
                    except:
                        print(f"Warning: Could not fetch config from {config_id}.")
                        # Try to find config in the same dir if possible, or raise
                        raise ValueError(f"Failed to load config for Qwen. Internet access required for {config_id}")

                    # 3. Instantiate Model
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        None, 
                        config=config, 
                        state_dict=state_dict, 
                        torch_dtype=torch.bfloat16, 
                        device_map="auto"
                    )
                    
                    # 4. Processor
                    processor = AutoProcessor.from_pretrained(config_id)
                    
                else:
                    # Directory strategy
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        full_path, torch_dtype=torch.bfloat16, device_map="auto"
                    )
                    processor = AutoProcessor.from_pretrained(full_path)

                set_cached_model("clip", key, {"model": model, "processor": processor})
                
            except Exception as e:
                print(f"Failed to load Qwen: {e}")
                raise
            
        return VNCCSCLIPOutput(clip_key=key)

# --- 6. Load VAE ---
@invocation("vnccs_load_vae", title="VNCCS Load VAE", tags=["vnccs", "vae"], category="vnccs", version="1.0.3")
class VNCCSLoadVAEInvocation(BaseInvocation):
    """Loads VAE model"""
    model_path: SDXLModelsLiteral = InputField(default=SDXL_MODELS[0] if SDXL_MODELS else "None", description="Select Model from SDXL Directory")
    
    def invoke(self, context: InvocationContext) -> VNCCSVAEOutput:
        from safetensors.torch import load_file
        import comfy.sd
        import comfy.utils
        
        full_path = os.path.join(SDXL_DIR, self.model_path)
        key = full_path
        
        if not get_cached_model("vae", key):
            print(f"Loading VAE from {full_path}...")
            try:
                if os.path.isfile(full_path):
                    # Single file loading strategy (ComfyUI Style)
                    print("Detected single file VAE. Using ComfyUI loading strategy...")
                    
                    # 1. Load state dict
                    sd = comfy.utils.load_torch_file(full_path)
                    
                    # 2. Instantiate VAE using ComfyUI's VAE wrapper
                    # This automatically infers config from state dict keys
                    vae = comfy.sd.VAE(sd=sd)
                    
                    # 3. Store the ComfyUI VAE object
                    # Note: Downstream nodes must support this object or we need to extract the model
                    set_cached_model("vae", key, vae)
                    
                else:
                    # Directory strategy (Fallback to diffusers or Comfy logic)
                    # If it's a directory, Comfy usually expects a checkpoint, but if it's a Diffusers folder:
                    from diffusers import AutoencoderKL
                    vae = AutoencoderKL.from_pretrained(full_path, torch_dtype=torch.float16).to("cuda")
                    set_cached_model("vae", key, vae)
                
            except Exception as e:
                print(f"Failed to load VAE: {e}")
                raise
            
        return VNCCSVAEOutput(vae_key=key)

# --- 7. VNCCS QWEN Encoder (The Core Logic) ---
@invocation("vnccs_qwen_encoder", title="VNCCS Qwen Encoder", tags=["vnccs", "qwen", "encoder"], category="vnccs", version="1.0.1")
class VNCCSQwenEncoderInvocation(BaseInvocation):
    """
    Encodes Image and Text using Qwen-VL for Conditioning.
    Matches ComfyUI VNCCS_QWEN_Encoder functionality.
    """
    clip_key: str = InputField(description="The loaded Qwen/CLIP model key")
    vae_key: str = InputField(description="VAE Key for reference latents")
    
    prompt: str = InputField(default="", description="User Prompt", ui_component=UIComponent.Textarea)
    instruction: str = InputField(default="Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.", description="System Instruction", ui_component=UIComponent.Textarea)
    
    image1: Optional[ImageField] = InputField(default=None, description="Image 1")
    image2: Optional[ImageField] = InputField(default=None, description="Image 2")
    image3: Optional[ImageField] = InputField(default=None, description="Image 3")
    
    weight1: float = InputField(default=1.0, description="Weight for Image 1")
    weight2: float = InputField(default=1.0, description="Weight for Image 2")
    weight3: float = InputField(default=1.0, description="Weight for Image 3")
    
    vl_size: int = InputField(default=384, description="Vision Language Size (for Qwen)")
    target_size: int = InputField(default=1024, description="Target Size for Reference Latents")
    latent_image_index: int = InputField(default=1, description="Which image to output as latent (1-based)")
    
    def invoke(self, context: InvocationContext) -> VNCCSConditioningOutput:
        import math
        import numpy as np
        
        clip_data = get_cached_model("clip", self.clip_key)
        vae = get_cached_model("vae", self.vae_key)
        
        if not clip_data or not vae:
            raise ValueError("Missing CLIP/Qwen or VAE model")
            
        model = clip_data["model"]
        processor = clip_data["processor"]
        
        # Prepare Images
        images_info = [
            (self.image1, self.weight1, "Picture 1"),
            (self.image2, self.weight2, "Picture 2"),
            (self.image3, self.weight3, "Picture 3")
        ]
        
        ref_latents = []
        vl_images = []
        conversation_content = [] # For processor
        
        # Helper for image processing
        def process_image(image_field, target_dim, is_vl=False):
            img_pil = context.images.get_pil(image_field.image_name)
            # Resize logic (Simple approximation of ComfyUI's common_upscale)
            # Calculate new size maintaining aspect ratio
            w, h = img_pil.size
            scale = math.sqrt((target_dim * target_dim) / (w * h))
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Ensure divisible by 8 (standard VAE requirement, Qwen might differ but safe)
            if not is_vl:
                new_w = math.ceil(new_w / 8) * 8
                new_h = math.ceil(new_h / 8) * 8
            
            img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return img_resized

        # Processing loop
        for img_field, weight, name in images_info:
            if img_field:
                # 1. Reference Latents (VAE)
                # Resize to target_size
                img_vae_pil = process_image(img_field, self.target_size, is_vl=False)
                
                # Convert to tensor [B, H, W, C] 0-1
                img_tensor = torch.from_numpy(np.array(img_vae_pil)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to("cuda") # Assuming GPU
                
                # Encode with VAE
                # ComfyUI VAE encode returns a dict or tensor depending on implementation
                # Using standard Comfy logic: vae.encode(pixels)
                with torch.no_grad():
                     latent = vae.encode(img_tensor[:,:,:,:3]) # Ensure 3 channels
                     # Apply weight (squared as per ComfyUI node)
                     weighted_latent = latent * (weight ** 2)
                     ref_latents.append(weighted_latent)
                
                # 2. Qwen Inputs (VL)
                # Resize to vl_size (optional but good for consistency)
                # Qwen2.5-VL handles variable sizes, but let's stick to user intent
                # ComfyUI node resizes if vl_resize is True (it is hardcoded True in source)
                img_vl_pil = process_image(img_field, self.vl_size, is_vl=True)
                vl_images.append(img_vl_pil)
                
                # Add to conversation
                conversation_content.append({"type": "text", "text": f"{name}: "})
                conversation_content.append({"type": "image", "image": img_vl_pil})

        # Construct Conversation
        # System instruction
        # User: [Images...] Prompt
        conversation_content.append({"type": "text", "text": self.prompt})
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.instruction}]
            },
            {
                "role": "user",
                "content": conversation_content
            }
        ]
        
        # Prepare Qwen Inputs
        print("Encoding with Qwen...")
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=vl_images,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use last hidden state (or appropriate layer)
            hidden_states = outputs.hidden_states[-1]
            
        # Create Conditioning Data Structure
        # We simulate the ComfyUI style: list of (tensor, dict)
        # But for InvokeAI cache, we store the dict directly
        
        # Filter zero-weight latents logic from ComfyUI:
        # ref_latents_full = [latent for latent, w in zip(ref_latents_weighted, weights_list) if w > 0]
        # Since we already weighted them and appended only if image exists, we assume they are valid.
        
        cond_data = {
            "embeddings": hidden_states,
            "reference_latents": ref_latents,
            "reference_latents_method": "index_timestep_zero" # Hardcoded default from ComfyUI node
        }
        
        cond_key = f"cond_qwen_{context.graph_execution_state_id}"
        set_cached_model("conditioning", cond_key, cond_data)
        
        # Prepare Latent Output (from latent_image_index)
        latent_out_key = f"latent_qwen_{context.graph_execution_state_id}"
        if len(ref_latents) >= self.latent_image_index:
             # ComfyUI node returns unweighted latent here?
             # Code says: samples = ref_latents[latent_image_index - 1]
             # But ref_latents in the list were weighted! 
             # Wait, in ComfyUI code: 
             # ref_latents_weighted = [ (w ** 2) * latent ... ]
             # ref_latents (original) were appended before weighting.
             # So I should probably store unweighted latents for output if needed.
             # But for simplicity, let's just use the computed one for now or fix logic.
             # Fix: Comfy keeps raw ref_latents separate.
             
             # Re-encoding the selected image without weight for output
             selected_img_info = images_info[self.latent_image_index - 1]
             if selected_img_info[0]:
                 img_vae_pil = process_image(selected_img_info[0], self.target_size, is_vl=False)
                 img_tensor = torch.from_numpy(np.array(img_vae_pil)).float() / 255.0
                 img_tensor = img_tensor.unsqueeze(0).to("cuda")
                 with torch.no_grad():
                    output_latent = vae.encode(img_tensor[:,:,:,:3])
                 set_cached_model("latents", latent_out_key, output_latent)
             else:
                 set_cached_model("latents", latent_out_key, torch.zeros(1, 4, 128, 128))
        else:
             set_cached_model("latents", latent_out_key, torch.zeros(1, 4, 128, 128))

        return VNCCSConditioningOutput(
            positive_conditioning=cond_key,
            negative_conditioning="" # Placeholder
        )

# --- 8. KSampler ---
@invocation("vnccs_ksampler", title="VNCCS KSampler", tags=["vnccs", "sampler"], category="vnccs", version="1.0.0")
class VNCCSKSamplerInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Custom Sampler for AuraFlow/Qwen pipeline"""
    unet_key: str = InputField(description="UNet Model Key (patched or original)")
    positive_conditioning: str = InputField(description="Positive Conditioning Key")
    negative_conditioning: str = InputField(description="Negative Conditioning Key")
    vae_key: str = InputField(description="VAE Key (for latent shape info)")
    seed: int = InputField(default=0, description="Random Seed")
    steps: int = InputField(default=20, description="Steps")
    cfg: float = InputField(default=3.5, description="CFG Scale")
    
    def invoke(self, context: InvocationContext) -> VNCCSLatentOutput:
        unet_wrapper = get_cached_model("unet", self.unet_key)
        pos_cond = get_cached_model("conditioning", self.positive_conditioning)
        
        if not unet_wrapper or pos_cond is None:
            raise ValueError("Missing model or conditioning")
        
        # Unwrap model and config
        unet_wrapper = wrap_model(unet_wrapper)
        model = unet_wrapper["model"]
        config = unet_wrapper["config"]
        
        # Extract patched parameters
        shift_val = config.get("shift", 1.0) # Default shift if not patched
        cfg_rescale_val = config.get("cfg_rescale", 0.0) # Default 0 if not patched
            
        print(f"Sampling with AuraFlow... Steps: {self.steps}, CFG: {self.cfg}")
        print(f"  > Active Patches: Shift={shift_val}, CFG_Rescale={cfg_rescale_val}")
        
        # Mock Sampling Loop (Replace with actual flow matching loop)
        # latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.bfloat16)
        # for i in range(self.steps):
        #     latents = scheduler.step(model_output, i, latents)
        
        # Creating a dummy latent result
        latents = torch.randn((1, 4, 128, 128), device="cuda") # Mock size
        
        latent_key = f"latent_{self.id}"
        set_cached_model("latents", latent_key, latents)
        
        return VNCCSLatentOutput(latents=latent_key)

# --- 9. VAE Decode ---
@invocation("vnccs_vae_decode", title="VNCCS VAE Decode", tags=["vnccs", "vae", "decode"], category="vnccs", version="1.0.1")
class VNCCSVAEDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decodes latents to Image"""
    vae_key: str = InputField(description="VAE Key")
    latents: str = InputField(description="Latents Key")
    
    def invoke(self, context: InvocationContext) -> VNCCSImageOutput:
        vae = get_cached_model("vae", self.vae_key)
        latents = get_cached_model("latents", self.latents)
        
        if not vae or latents is None:
            raise ValueError("Missing VAE or Latents")
            
        print("Decoding latents...")
        with torch.no_grad():
            # Check if it's a ComfyUI VAE object (has 'decode' method but check signature/type)
            # ComfyUI VAE class has a decode method that takes tensor
            try:
                # ComfyUI VAE decode wrapper
                # It handles scaling and processing internally usually?
                # ComfyUI VAE.decode returns the image tensor
                image = vae.decode(latents)
            except Exception:
                 # Fallback for Diffusers VAE (if previously loaded via diffusers)
                 # Diffusers VAE decode returns DecoderOutput with .sample
                 try:
                     image = vae.decode(latents / vae.config.scaling_factor).sample
                 except:
                     # Raw decode attempt
                     image = vae.decode(latents)

        # Post-process image (Standardize to HWC, uint8, 0-255)
        # ComfyUI VAE decode output is usually (B, H, W, C) or (B, C, H, W)?
        # ComfyUI VAE.decode returns (Batch, Height, Width, Channels) ? No, usually (B, C, H, W) in pytorch
        # Let's check dimensions
        
        if len(image.shape) == 4:
            if image.shape[1] == 3: # BCHW
                image = image.permute(0, 2, 3, 1) # to BHWC
        
        image = (image / 2 + 0.5).clamp(0, 1) if image.min() < 0 else image.clamp(0, 1)
        # Note: ComfyUI VAE.process_output already does (image + 1.0) / 2.0 and clamps if used
        # If we double dip, it might be washed out.
        # But safely clamping 0-1 is fine.
        
        image = image.float().cpu().numpy()
        image = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(image[0])
        
        # Save to InvokeAI Gallery
        image_dto = context.images.save(image=pil_image)
        
        return VNCCSImageOutput(image=ImageField(image_name=image_dto.image_name))
