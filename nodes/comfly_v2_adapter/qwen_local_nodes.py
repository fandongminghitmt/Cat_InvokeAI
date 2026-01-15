from typing import Literal, Optional, List
import torch
from PIL import Image
import io
import os

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
    OutputField
)

# Global cache for the model to avoid reloading on every generation
# In a production environment, this should be managed by a proper ModelManager
_QWEN_MODEL_CACHE = {
    "model": None,
    "processor": None,
    "model_path": None
}

@invocation_output("qwen_local_output")
class QwenLocalOutput(BaseInvocationOutput):
    """Output for Qwen Local Invocation"""
    generated_text: str = OutputField(description="The generated text/prompt from Qwen")
    positive_prompt: str = OutputField(description="Processed positive prompt for SD/Flux")
    negative_prompt: str = OutputField(description="Processed negative prompt for SD/Flux")

@invocation(
    "qwen_local_generator",
    title="Qwen2.5-VL Local Generator",
    tags=["qwen", "vl", "local", "ai"],
    category="local_models",
    version="1.0.0"
)
class QwenLocalGeneratorInvocation(BaseInvocation, WithMetadata, WithBoard):
    """
    Locally loads Qwen2.5-VL model to process images and instructions.
    Requires the model to be downloaded locally.
    """
    
    image: ImageField = InputField(description="Input image for the model to analyze")
    model_path: str = InputField(
        default="Qwen/Qwen2.5-VL-7B-Instruct", 
        description="Local path to the Qwen2.5-VL model directory or HuggingFace ID"
    )
    instruction: str = InputField(
        default="Describe this image in detail for an image generation prompt.", 
        description="Instruction for the model"
    )
    precision: Literal["bf16", "fp16", "fp32"] = InputField(
        default="bf16", 
        description="Model precision to load"
    )
    
    def invoke(self, context: InvocationContext) -> QwenLocalOutput:
        global _QWEN_MODEL_CACHE
        
        # Import transformers here
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("Please install transformers>=4.49.0 to use Qwen2.5-VL")

        # 1. Load Image
        image_pil = context.images.get_pil(self.image.image_name)
        
        # 2. Load Model (Singleton Pattern)
        if _QWEN_MODEL_CACHE["model"] is None or _QWEN_MODEL_CACHE["model_path"] != self.model_path:
            print(f"Loading Qwen model from {self.model_path}...")
            
            torch_dtype = torch.bfloat16 if self.precision == "bf16" else (torch.float16 if self.precision == "fp16" else torch.float32)
            
            # Load model
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            
            # Load processor
            processor = AutoProcessor.from_pretrained(self.model_path)
            
            _QWEN_MODEL_CACHE["model"] = model
            _QWEN_MODEL_CACHE["processor"] = processor
            _QWEN_MODEL_CACHE["model_path"] = self.model_path
            print("Qwen model loaded successfully.")
            
        model = _QWEN_MODEL_CACHE["model"]
        processor = _QWEN_MODEL_CACHE["processor"]
        
        # 3. Prepare Inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": self.instruction},
                ],
            }
        ]
        
        # Preprocessing
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages) # Need to import this or implement
        
        # Simplification: Let processor handle image directly if supported, or manually construct
        inputs = processor(
            text=[text],
            images=[image_pil],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda") # Assuming CUDA
        
        # 4. Generate
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # 5. Return Result
        return QwenLocalOutput(
            generated_text=output_text,
            positive_prompt=output_text, 
            negative_prompt="low quality, bad anatomy, distortion"
        )
