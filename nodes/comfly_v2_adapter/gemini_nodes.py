import os
import json
import time
import logging
import requests
import base64
import io
import urllib3
import math
from PIL import Image
from typing import Optional, Literal, List
from io import BytesIO

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, InvocationContext, BaseInvocationOutput, invocation_output
from invokeai.app.invocations.fields import InputField, ImageField, Input, OutputField, UIType, WithBoard, WithMetadata, BoardField
from invokeai.app.invocations.primitives import ImageOutput, StringOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from .utils import base64_to_pil, pil_to_base64, find_or_create_board

def get_node_config():
    """
    Returns a dict containing 'api_key' and 'base_url' from Comflyapi.json
    """
    config_data = {'api_key': '', 'base_url': ''}
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded = json.load(f)
                config_data['api_key'] = loaded.get('api_key', '')
                config_data['base_url'] = loaded.get('base_url', '')
    except:
        pass
    return config_data

@invocation_output("gemini_node_output")
class GeminiNodeOutput(BaseInvocationOutput):
    """Output for Gemini Node containing both image and text"""
    image: ImageField = OutputField(description="The generated image")
    text: str = OutputField(description="The generated text")
    width: int = OutputField(description="Image width")
    height: int = OutputField(description="Image height")

@invocation("comfly_gemini_nano_banana", title="Comfly Nano Banana Pro (Gemini)", tags=["gemini", "image", "gen", "google"], category="comfly", version="1.0.0")
class ComflyGeminiNanoBananaInvocation(BaseInvocation, WithMetadata):
    """Generate or edit images using Google Gemini/Banana via Comfly API (Image Endpoints)"""
    
    use_board: Optional[str] = InputField(default=None, description="Board ID", input=Input.Direct)
    prompt: str = InputField(description="Text prompt describing the image to generate or edit", ui_component="textarea")
    model: Literal["gemini-3-pro-image-preview", "gemini-2.5-flash-image", "nano-banana-2", "nano-banana", "nano-banana-hd"] = InputField(default="nano-banana-2", description="Model to use")
    api_key: str = InputField(default="", description="API Key (Leave empty to use config)")
    
    input_image: Optional[ImageField] = InputField(description="Optional reference image 1", input=Input.Connection, default=None)
    image_2: Optional[ImageField] = InputField(description="Input image 2", input=Input.Connection, default=None)
    image_3: Optional[ImageField] = InputField(description="Input image 3", input=Input.Connection, default=None)
    image_4: Optional[ImageField] = InputField(description="Input image 4", input=Input.Connection, default=None)
    
    seed: int = InputField(default=42, description="Seed (Control after generate)")
    
    aspect_ratio: Literal["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"] = InputField(default="1:1", description="Aspect Ratio")
    resolution: Literal["1K", "2K", "4K"] = InputField(default="1K", description="Target resolution (Only for Banana-2)")
    
    base_url: str = InputField(default="https://ai.comfly.chat", description="Base URL for API")

    def invoke(self, context: InvocationContext) -> GeminiNodeOutput:
        logger = logging.getLogger("InvokeAI")
        start_total = time.time()
        
        msg_start = f"[ComflyBanana] Starting execution with model: {self.model}"
        logger.info(msg_start)
        print(msg_start)
        
        # Load Config
        node_config = get_node_config()
        
        # API Key
        key_to_use = self.api_key.strip()
        if not key_to_use:
            key_to_use = node_config['api_key']
        if not key_to_use:
            raise Exception("API Key is missing.")

        # Base URL
        url_to_use = self.base_url.strip()
        default_url = "https://ai.comfly.chat"
        if url_to_use == default_url and node_config['base_url']:
            url_to_use = node_config['base_url']
        url_to_use = url_to_use.rstrip('/')

        # Session Setup
        session = requests.Session()
        session.trust_env = False
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry_strategy = Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Collect Images
        image_inputs = [self.input_image, self.image_2, self.image_3, self.image_4]
        valid_images = [img for img in image_inputs if img is not None]
        has_images = len(valid_images) > 0

        # Determine Mode and Endpoint
        mode = "img2img" if has_images else "text2img"
            
        # Headers
        headers = {
            "Authorization": f"Bearer {key_to_use}"
        }

        # Construct Request
        response = None
        
        # Query Params for Async
        query_params = {"async": "true"}

        try:
            if mode == "text2img":
                api_endpoint = f"{url_to_use}/v1/images/generations"
                headers["Content-Type"] = "application/json"
                
                payload = {
                    "prompt": self.prompt,
                    "model": self.model,
                    "response_format": "b64_json"
                }
                
                # Aspect Ratio handling
                if self.aspect_ratio != "auto":
                    payload["aspect_ratio"] = self.aspect_ratio
                
                # Image Size handling (Specific to nano-banana-2)
                if self.model == "nano-banana-2":
                    payload["image_size"] = self.resolution  # "1K", "2K", "4K"
                
                if self.seed > 0:
                    payload["seed"] = self.seed

                t_before_req = time.time()
                print(f"[ComflyBanana] Sending Text2Img request to {api_endpoint}")
                print(f"[ComflyBanana] Payload params: model={self.model}, ar={self.aspect_ratio}, size={payload.get('image_size', 'N/A')}")
                
                response = session.post(
                    api_endpoint,
                    headers=headers,
                    json=payload,
                    params=query_params,
                    timeout=120
                )
                
            else: # img2img / edit
                api_endpoint = f"{url_to_use}/v1/images/edits"
                
                files = []
                for i, img_field in enumerate(valid_images):
                    pil_image = context._services.images.get_pil_image(img_field.image_name)
                    # Convert to PNG
                    buffered = BytesIO()
                    pil_image.save(buffered, format="PNG")
                    buffered.seek(0)
                    files.append(('image', (f'image_{i}.png', buffered, 'image/png')))
                
                data = {
                    "prompt": str(self.prompt),
                    "model": str(self.model),
                    "response_format": "b64_json"
                }
                
                if self.aspect_ratio != "auto":
                    data["aspect_ratio"] = str(self.aspect_ratio)
                
                # Image Size handling (Specific to nano-banana-2)
                if self.model == "nano-banana-2":
                    data["image_size"] = str(self.resolution)

                if self.seed > 0:
                    data["seed"] = str(self.seed)
                    
                t_before_req = time.time()
                print(f"[ComflyBanana] Sending Img2Img request to {api_endpoint} with {len(files)} images")
                print(f"[ComflyBanana] Form Data: {data}")

                response = session.post(
                    api_endpoint,
                    headers=headers, 
                    data=data,       
                    files=files,     
                    params=query_params,
                    timeout=300
                )

            # Handle Response (Async Task ID)
            print(f"[ComflyBanana] Request took: {time.time() - t_before_req:.2f}s")
            if response.status_code != 200:
                raise Exception(f"API Error {response.status_code}: {response.text}")
            
            result = response.json()
            task_id = result.get("task_id") or result.get("data") 
            
            if not task_id:
                if "data" in result and isinstance(result["data"], list):
                     print("[ComflyBanana] Sync response received.")
                     return self._process_image_result(context, result, start_total)
                raise Exception(f"No task_id returned: {result}")
                
            print(f"[ComflyBanana] Task submitted. ID: {task_id}")
            
            # Poll for completion
            max_retries = 600
            retry_count = 0
            
            while retry_count < max_retries:
                time.sleep(2)
                retry_count += 1
                
                status_url = f"{url_to_use}/v1/images/tasks/{task_id}"
                
                t_poll = time.time()
                status_resp = session.get(status_url, headers=headers, timeout=30)
                print(f"[ComflyBanana] Poll check took: {time.time() - t_poll:.2f}s")
                
                if status_resp.status_code != 200:
                    print(f"Status check failed: {status_resp.status_code}")
                    continue
                
                status_json = status_resp.json()
                
                outer_data = status_json.get("data", {})
                status = outer_data.get("status", "")
                
                if status == "SUCCESS":
                    print("[ComflyBanana] Task SUCCESS.")
                    inner_data = outer_data.get("data", {})
                    return self._process_image_result(context, inner_data, start_total)
                    
                elif status == "FAILURE":
                    fail_reason = outer_data.get("fail_reason", "Unknown")
                    raise Exception(f"Task Failed: {fail_reason}")
                
                if retry_count % 5 == 0:
                    print(f"[ComflyBanana] Waiting... ({retry_count}/{max_retries}) Status: {status}")

            raise Exception("Timeout waiting for task completion.")

        except Exception as e:
            logger.error(f"[ComflyBanana] Error: {e}")
            raise e

    def _process_image_result(self, context, result_data, start_time):
        image_list = result_data.get("data", [])
        if not image_list:
             if isinstance(result_data, list):
                 image_list = result_data
             else:
                 raise Exception(f"No image data found in result: {result_data}")
        
        first_image = image_list[0]
        pil_image = None
        
        if "b64_json" in first_image:
            img_bytes = base64.b64decode(first_image["b64_json"])
            pil_image = Image.open(io.BytesIO(img_bytes))
        elif "url" in first_image:
            import requests
            img_url = first_image["url"]
            print(f"[ComflyBanana] Downloading image from {img_url}")
            t_dl = time.time()
            resp = requests.get(img_url, verify=False)
            print(f"[ComflyBanana] Download took: {time.time() - t_dl:.2f}s")
            pil_image = Image.open(io.BytesIO(resp.content))
        else:
            raise Exception("Unknown image format in response")
            
        # Force image to be non-intermediate
        self.is_intermediate = False
        if hasattr(context._data.invocation, 'is_intermediate'):
            context._data.invocation.is_intermediate = False

        # Explicitly use the board selected by the user
        board_id = find_or_create_board(context, self.use_board)

        # Use context.images.save for consistency with Flux node
        image_dto = context.images.save(image=pil_image, board_id=board_id)
        
        print(f"[ComflyBanana] Total time: {time.time() - start_time:.2f}s")
        
        return GeminiNodeOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
            text="Generated by Comfly Nano Banana"
        )
