import os
import json
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, InvocationContext
from invokeai.app.invocations.fields import InputField, ImageField, FieldDescriptions, Input, OutputField, UIType, WithBoard, WithMetadata, BoardField
from typing import Optional
from invokeai.app.invocations.primitives import ImageOutput, StringOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from PIL import Image
import requests
import base64
import io
from .utils import base64_to_pil, find_or_create_board

def get_config_api_key():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('api_key', '')
    except:
        pass
    return ''

@invocation("comfly_openai_gen_image", title="Comfly OpenAI Image Gen", tags=["openai", "image", "gen"], category="comfly", version="1.0.0")
class ComflyOpenAIGenImageInvocation(BaseInvocation, WithMetadata):
    """Generate images using OpenAI DALL-E models via Comfly logic"""
    
    use_board: Optional[str] = InputField(default=None, description="Board ID", input=Input.Direct)
    prompt: str = InputField(description="Image prompt")
    api_key: str = InputField(default="", description="OpenAI API Key (Leave empty to use config)")
    model: str = InputField(default="dall-e-3", description="Model to use (e.g. dall-e-3, dall-e-2)")
    size: str = InputField(default="1024x1024", description="Image size")
    quality: str = InputField(default="standard", description="Quality (standard/hd)")
    base_url: str = InputField(default="https://ai.comfly.chat", description="Base URL for API")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Use provided API key or fallback to config
        key_to_use = self.api_key.strip()
        if not key_to_use:
            key_to_use = get_config_api_key()
            
        if not key_to_use:
            raise Exception("API Key is missing. Please provide it in the node or set it via Comfly API Config node.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key_to_use}"
        }
        
        payload = {
            "prompt": self.prompt,
            "model": self.model,
            "n": 1,
            "size": self.size,
            "quality": self.quality,
            "response_format": "b64_json"
        }
        
        url = f"{self.base_url.rstrip('/')}/v1/images/generations"
        url = url.strip()
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise Exception(f"API Error: {response.text}")
            
        result = response.json()
        if "data" not in result or not result["data"]:
            raise Exception("No data in response")
            
        b64_data = result["data"][0]["b64_json"]
        image = base64_to_pil(b64_data)
        
        # Use internal service to control is_intermediate. Default to False (always show in gallery)
        # since we removed the explicit checkbox.
        # Explicitly use the board selected by the user
        
        # Force image to be non-intermediate (visible in Gallery)
        try:
            context._data.invocation.is_intermediate = False
        except:
            pass

        board_id = find_or_create_board(context, self.use_board)
        image_dto = context.images.save(image=image, board_id=board_id)
        
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )

@invocation("comfly_openai_chat", title="Comfly OpenAI Chat", tags=["openai", "chat", "vision"], category="comfly", version="1.0.0")
class ComflyOpenAIChatInvocation(BaseInvocation):
    """Chat with OpenAI models (supports Vision)"""
    
    prompt: str = InputField(description="User prompt")
    api_key: str = InputField(default="", description="OpenAI API Key (Leave empty to use config)")
    image: Optional[ImageField] = InputField(description="Input image for Vision models", input=Input.Connection, default=None)
    model: str = InputField(default="gpt-4-vision-preview", description="Model name")
    base_url: str = InputField(default="https://ai.comfly.chat", description="Base URL")

    def invoke(self, context: InvocationContext) -> StringOutput:
        # Use provided API key or fallback to config
        key_to_use = self.api_key.strip()
        if not key_to_use:
            key_to_use = get_config_api_key()
            
        if not key_to_use:
            raise Exception("API Key is missing. Please provide it in the node or set it via Comfly API Config node.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key_to_use}"
        }
        
        messages = []
        content = [{"type": "text", "text": self.prompt}]
        
        if self.image:
            pil_image = context.services.images.get_pil_image(self.image.image_name)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }
            })
            
        messages.append({"role": "user", "content": content})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096
        }
        
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        url = url.strip()
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            raise Exception(f"API Error: {response.text}")
            
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        return StringOutput(value=content)
