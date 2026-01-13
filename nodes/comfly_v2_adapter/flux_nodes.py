import os
import json
import time
import logging
import requests
import base64
import io
import urllib3
from PIL import Image
from typing import Optional, Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, InvocationContext
from invokeai.app.invocations.fields import InputField, ImageField, Input, OutputField, WithBoard, WithMetadata, BoardField
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

def get_config_api_key():
    # Backwards compatibility helper if needed, but we'll use get_node_config mostly
    return get_node_config()['api_key']

@invocation("comfly_flux_2_max", title="Comfly Flux 2 Max", tags=["flux", "image", "gen", "max"], category="comfly", version="1.0.0")
class ComflyFlux2MaxInvocation(BaseInvocation, WithMetadata):
    """Generate images using Flux 2 Max (via Comfly/BFL API)"""
    
    use_board: Optional[str] = InputField(default=None, description="Board ID", input=Input.Direct)
    prompt: str = InputField(description="Image prompt", ui_component="textarea")
    model: Literal["flux-kontext-max", "flux-kontext-pro", "flux-2-max"] = InputField(default="flux-kontext-max", description="Model to use")
    api_key: str = InputField(default="", description="API Key (Leave empty to use config)")
    
    input_image: Optional[ImageField] = InputField(description="Input image 1", input=Input.Connection, default=None)
    image_2: Optional[ImageField] = InputField(description="Input image 2", input=Input.Connection, default=None)
    image_3: Optional[ImageField] = InputField(description="Input image 3", input=Input.Connection, default=None)
    image_4: Optional[ImageField] = InputField(description="Input image 4", input=Input.Connection, default=None)
    image_5: Optional[ImageField] = InputField(description="Input image 5", input=Input.Connection, default=None)
    image_6: Optional[ImageField] = InputField(description="Input image 6", input=Input.Connection, default=None)
    image_7: Optional[ImageField] = InputField(description="Input image 7", input=Input.Connection, default=None)
    image_8: Optional[ImageField] = InputField(description="Input image 8", input=Input.Connection, default=None)
    image_9: Optional[ImageField] = InputField(description="Input image 9", input=Input.Connection, default=None)
    
    seed: int = InputField(default=-1, description="Seed (-1 for random)")
    
    width: int = InputField(default=1024, description="Width (for Flux 2 Pro/Flex/Max)")
    height: int = InputField(default=1024, description="Height (for Flux 2 Pro/Flex/Max)")
    
    aspect_ratio: Literal["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"] = InputField(default="1:1", description="Aspect Ratio (for Kontext models)")
    output_format: Literal["png", "jpeg"] = InputField(default="png", description="Output format")
    prompt_upsampling: bool = InputField(default=False, description="Enable prompt upsampling")
    safety_tolerance: int = InputField(default=2, ge=0, le=6, description="Safety tolerance (0-6)")
    
    base_url: str = InputField(default="https://ai.comfly.chat", description="Base URL for API")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        logger = logging.getLogger("InvokeAI")
        # Start tracking total time
        start_total = time.time()
        logger.info(f"[ComflyFlux2Max] Starting execution...")

        # Load Config
        node_config = get_node_config()
        
        # 1. API Key Handling
        key_to_use = self.api_key.strip()
        if not key_to_use:
            key_to_use = node_config['api_key']
        if not key_to_use:
            raise Exception("API Key is missing. Please provide it in the node or set it via Comfly API Config node.")

        # Determine Base URL
        url_to_use = self.base_url.strip()
        default_url = "https://ai.comfly.chat"
        if url_to_use == default_url and node_config['base_url']:
            url_to_use = node_config['base_url']
            
        # Ensure no trailing slash
        url_to_use = url_to_use.rstrip('/')

        # Setup Session with trust_env=False to bypass system proxy issues
        session = requests.Session()
        session.trust_env = False
        session.verify = False  # Ignore SSL certificate verification to avoid SSLEOFError
        
        # Force HTTP adapter to bypass complex SSL issues
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Suppress insecure request warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key_to_use}"
        }

        # 2. Prepare Payload
        payload = {
            "prompt": self.prompt,
            "output_format": self.output_format,
            "safety_tolerance": self.safety_tolerance
        }

        # Handle prompt_upsampling
        if self.prompt_upsampling:
            payload["prompt_upsampling"] = self.prompt_upsampling

        # Handle Image Inputs
        image_inputs = [
            ("input_image", self.input_image),
            ("image_2", self.image_2),
            ("image_3", self.image_3),
            ("image_4", self.image_4),
            ("image_5", self.image_5),
            ("image_6", self.image_6),
            ("image_7", self.image_7),
            ("image_8", self.image_8),
            ("image_9", self.image_9),
        ]

        has_any_image = False
        for param_name, img_field in image_inputs:
            if img_field:
                has_any_image = True
                pil_image = context._services.images.get_pil_image(img_field.image_name)
                
                # Optimize image size to avoid 413 Payload Too Large
                # Convert to RGB if needed (for JPEG)
                if pil_image.mode == 'RGBA':
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[3])
                    pil_image = background
                elif pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Resize if too large (max 2048 on long edge)
                max_dim = 2048
                if max(pil_image.width, pil_image.height) > max_dim:
                    pil_image.thumbnail((max_dim, max_dim), Image.LANCZOS)
                
                # Use JPEG instead of PNG for compression
                img_base64 = pil_to_base64(pil_image, format="JPEG")
                
                if img_base64:
                    api_key_name = param_name
                    if param_name != "input_image":
                        api_key_name = "input_" + param_name
                    payload[api_key_name] = img_base64

        logger.info(f"[ComflyFlux2Max] Image preparation took: {time.time() - start_total:.2f}s")

        # Handle Dimensions vs Aspect Ratio based on model type
        if "kontext" in self.model:
            # Kontext models use aspect_ratio
            if self.aspect_ratio != "Default":
                 payload["aspect_ratio"] = self.aspect_ratio
        else:
            # Other models (Flux 2 Pro/Max/etc) use width/height
            if self.width > 0 and self.height > 0:
                 payload["width"] = self.width
                 payload["height"] = self.height

        # Handle Seed
        if self.seed != -1:
            payload["seed"] = self.seed

        # 3. Submit Task
        api_endpoint = f"{url_to_use}/bfl/v1/{self.model}"
        
        response = session.post(
            api_endpoint,
            headers=headers,
            json=payload,
            timeout=300
        )

        if response.status_code != 200:
            error_msg = response.text
            try:
                # Try to parse JSON error message
                error_json = response.json()
                if "error" in error_json and "message" in error_json["error"]:
                    error_msg = error_json["error"]["message"]
            except:
                pass
                
            if response.status_code == 403 and "用户已被封禁" in error_msg:
                 raise Exception(f"API Error: Account Banned. Please contact support. Details: {error_msg}")
            
            if response.status_code == 401:
                 raise Exception(f"API Error: Invalid API Key. Please check your API key in the Comfly API Config node. Details: {error_msg}")

            raise Exception(f"API Error: {response.status_code} - {error_msg}")

        result = response.json()
        if "id" not in result:
            raise Exception(f"No task ID in response: {result}")

        task_id = result["id"]
        logger.info(f"[ComflyFlux2Max] API Submission took: {time.time() - start_total:.2f}s (cumulative). Task ID: {task_id}")
        
        # 4. Poll for Result
        max_attempts = 120  # Max wait time ~120s if polling every 1s, but actual time varies
        image_url = ""
        
        # Start with short interval, increase slightly if needed, but keep it snappy
        poll_interval = 0.5 
        
        t_before_poll = time.time()
        
        for attempt in range(max_attempts):
            # Dynamic sleep: shorter at first to catch fast completions
            time.sleep(poll_interval)
            
            # Gradually increase polling interval to reduce API load for long running tasks, but cap at 2s
            if attempt > 10:
                poll_interval = 1.0
            if attempt > 30:
                poll_interval = 2.0
            
            check_url = f"{url_to_use}/bfl/v1/get_result?id={task_id}"
            try:
                check_response = session.get(check_url, headers=headers, timeout=300)
                if check_response.status_code != 200:
                    continue
                
                check_data = check_response.json()
                status = check_data.get("status")
                
                if status == "Ready":
                    if "result" in check_data and "sample" in check_data["result"]:
                        image_url = check_data["result"]["sample"]
                        logger.info(f"[ComflyFlux2Max] Polling finished. Status: Ready. Time waiting: {time.time() - t_before_poll:.2f}s")
                        break
                elif status in ["Failed", "Error"]:
                    raise Exception(f"Task failed: {check_data.get('details', 'Unknown error')}")
                    
            except Exception as e:
                # Log error but continue polling unless it's a fatal error
                if "Task failed" in str(e):
                    raise e
                pass
        
        if not image_url:
            raise Exception("Timed out waiting for image generation")

        # 5. Download and Save Image
        t_before_download = time.time()
        img_response = session.get(image_url, timeout=300)
        img_response.raise_for_status()
        
        generated_image = Image.open(io.BytesIO(img_response.content))
        
        logger.info(f"[ComflyFlux2Max] Image download took: {time.time() - t_before_download:.2f}s")
        
        # Always save to gallery (is_intermediate=False)
        # Save Image
        # Explicitly use the board selected by the user
        
        # Force image to be non-intermediate (visible in Gallery)
        try:
            context._data.invocation.is_intermediate = False
        except:
            pass

        board_id = find_or_create_board(context, self.use_board)
        image_dto = context.images.save(image=generated_image, board_id=board_id)
        
        logger.info(f"[ComflyFlux2Max] Total execution time: {time.time() - start_total:.2f}s")
        
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
