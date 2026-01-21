import os
import json
import time
import logging
import requests
import base64
import io
import urllib3
import cv2
import numpy as np
import uuid
from PIL import Image
from typing import Optional, Literal, List, Any

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, InvocationContext, BaseInvocationOutput, invocation_output
from invokeai.app.invocations.fields import InputField, ImageField, Input, OutputField, UIType, WithBoard, WithMetadata, BoardField
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.invocation_context import InvocationContext
from .utils import pil_to_base64, find_or_create_board

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

@invocation_output("veo3_node_output")
class Veo3NodeOutput(BaseInvocationOutput):
    """Output for Google Veo3 Node containing video URL and response"""
    video_url: str = OutputField(description="The generated video URL")
    response_json: str = OutputField(description="The raw JSON response from the API")
    video_name: str = OutputField(description="The name of the video in the gallery")
    # We use ImageField here to trick the frontend into refreshing the gallery
    video: ImageField = OutputField(description="The generated video record")

@invocation("comfly_google_veo3", title="Comfly Google Veo3", tags=["google", "video", "veo3", "gen"], category="comfly", version="1.0.0")
class ComflyGoogleVeo3Invocation(BaseInvocation, WithMetadata):
    """Generate videos using Google Veo3 via Comfly API"""
    
    use_board: Optional[str] = InputField(default=None, description="Board ID", input=Input.Direct)
    prompt: str = InputField(description="Text prompt describing the video to generate", ui_component="textarea")
    model: Literal["veo3", "veo3-fast", "veo3-pro", "veo3-fast-frames", "veo3-pro-frames", "veo3.1", "veo3.1-pro", "veo3.1-components", "veo3.1-4k", "veo3.1-pro-4k", "veo3.1-components-4k"] = InputField(default="veo3", description="Model to use")
    enhance_prompt: bool = InputField(default=False, description="Enhance prompt using AI")
    aspect_ratio: Literal["16:9", "9:16"] = InputField(default="16:9", description="Aspect Ratio")
    
    api_key: str = InputField(default="", description="API Key (Leave empty to use config)")
    
    image_1: Optional[ImageField] = InputField(description="Optional reference image 1", input=Input.Connection, default=None)
    image_2: Optional[ImageField] = InputField(description="Optional reference image 2", input=Input.Connection, default=None)
    image_3: Optional[ImageField] = InputField(description="Optional reference image 3", input=Input.Connection, default=None)
    
    seed: int = InputField(default=0, description="Seed (0 for random)")
    enable_upsample: bool = InputField(default=False, description="Enable upsampling")
    
    base_url: str = InputField(default="https://ai.comfly.chat", description="Base URL for API")

    def invoke(self, context: InvocationContext) -> Veo3NodeOutput:
        logger = logging.getLogger("InvokeAI")
        start_total = time.time()
        
        msg_start = f"[ComflyVeo3] Starting execution..."
        logger.info(msg_start)
        print(msg_start)
        
        # Check for VideoService availability
        if not hasattr(context._services, 'videos'):
             error_msg = "[ComflyVeo3] Error: 'videos' service not found in context. This node requires VideoService to be enabled in InvokeAI."
             logger.error(error_msg)
             raise Exception(error_msg)

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

        # Setup Session
        session = requests.Session()
        session.trust_env = False
        session.verify = False  # Ignore SSL certificate verification
        
        # Force HTTP adapter
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key_to_use}"
        }

        # 2. Prepare Payload
        payload = {
            "prompt": self.prompt,
            "model": self.model,
            "enhance_prompt": self.enhance_prompt
        }

        if self.seed > 0:
            payload["seed"] = self.seed

        if self.model in ["veo3", "veo3-fast", "veo3-pro", "veo3.1", "veo3.1-pro", "veo3.1-components", "veo3.1-4k", "veo3.1-pro-4k", "veo3.1-components-4k"] and self.aspect_ratio:
            payload["aspect_ratio"] = self.aspect_ratio

        if self.model in ["veo3", "veo3-fast", "veo3-pro", "veo3.1", "veo3.1-pro", "veo3.1-components", "veo3.1-4k", "veo3.1-pro-4k", "veo3.1-components-4k"] and self.enable_upsample:
            payload["enable_upsample"] = self.enable_upsample

        # Handle Images
        images_base64 = []
        image_inputs = [self.image_1, self.image_2, self.image_3]
        
        for img_field in image_inputs:
            if img_field:
                pil_image = context._services.images.get_pil_image(img_field.image_name)
                # Convert to base64
                img_b64 = pil_to_base64(pil_image)
                if img_b64:
                    images_base64.append(f"data:image/png;base64,{img_b64}")
        
        if images_base64:
            payload["images"] = images_base64

        # 3. Submit Task
        api_endpoint = f"{url_to_use}/v2/videos/generations"
        
        msg_req = f"[ComflyVeo3] Sending request to {api_endpoint}"
        logger.info(msg_req)
        print(msg_req)
        
        try:
            response = session.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=300
            )
        except Exception as e:
            raise Exception(f"Connection error: {str(e)}")

        if response.status_code != 200:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            raise Exception(error_msg)

        result = response.json()
        
        task_id = result.get("task_id")
        if not task_id:
            # Fallback to old format just in case, or error out
            task_id = result.get("data")
        
        if not task_id:
            error_message = f"API Error: {result.get('message', 'No task ID returned')}"
            raise Exception(error_message)
            
        msg_task = f"[ComflyVeo3] Task ID: {task_id}. Waiting for completion..."
        logger.info(msg_task)
        print(msg_task)

        # 4. Polling for Result
        max_attempts = 600 
        attempts = 0
        video_url = None
        
        while attempts < max_attempts:
            time.sleep(2) 
            attempts += 1
            
            try:
                status_response = session.get(
                    f"{url_to_use}/v2/videos/generations/{task_id}",
                    headers=headers,
                    timeout=300
                )
                
                if status_response.status_code != 200:
                    continue
                    
                status_result = status_response.json()
                
                status = status_result.get("status", "")
                progress = status_result.get("progress", "0%")
                
                # Log progress periodically
                if attempts % 5 == 0:
                    logger.info(f"[ComflyVeo3] Status: {status}, Progress: {progress}")
                    print(f"[ComflyVeo3] Status: {status}, Progress: {progress}")
                
                if status == "SUCCESS":
                    data = status_result.get("data", {})
                    if "output" in data:
                        video_url = data["output"]
                        break
                    # Fallback for old API structure if needed, though unlikely with v2 endpoint
                    elif "data" in data and "video_url" in data["data"]:
                        video_url = data["data"]["video_url"]
                        break
                elif status == "FAILURE":
                    fail_reason = status_result.get("fail_reason", "Unknown error")
                    raise Exception(f"Video generation failed: {fail_reason}")
                    
            except Exception as e:
                # If checking status fails, we just retry unless it's a critical error
                if "Video generation failed" in str(e):
                    raise e
                print(f"Error checking generation status: {str(e)}")
        
        if not video_url:
            raise Exception("Failed to retrieve video URL after multiple attempts")
        
        msg_success = f"[ComflyVeo3] Video generated successfully: {video_url}"
        logger.info(msg_success)
        print(msg_success)

        # 5. Upload Video to Gallery (using VideoService)
        video_name = ""
        
        try:
            msg_dl = f"[ComflyVeo3] Downloading video to memory and uploading to Gallery..."
            logger.info(msg_dl)
            print(msg_dl)
            
            # Download to memory (BytesIO)
            vid_response = requests.get(video_url, stream=True)
            vid_response.raise_for_status()
            
            # We need to read it into a BytesIO object for the service
            video_bytes = vid_response.content
            
            # Use VideoService to create the record
            # We supply all required arguments: video_file, video_origin, video_category
            # Plus context info: node_id, session_id
            
            # Force to be non-intermediate
            self.is_intermediate = False
            if hasattr(context._data.invocation, 'is_intermediate'):
                context._data.invocation.is_intermediate = False
            
            video_dto = context._services.videos.create(
                video_file=video_bytes,
                video_origin=ResourceOrigin.INTERNAL,
                video_category=ImageCategory.GENERAL,
                node_id=context._data.invocation.id,
                session_id=context._data.queue_item.session_id,
                is_intermediate=False,
                board_id = find_or_create_board(context, self.use_board)
            )
            
            video_name = video_dto.video_name
            logger.info(f"[ComflyVeo3] Video uploaded to Gallery successfully: {video_name}")
            print(f"[ComflyVeo3] Video uploaded to Gallery successfully: {video_name}")

        except Exception as e:
            logger.error(f"[ComflyVeo3] Error uploading video to Gallery: {str(e)}")
            print(f"Error uploading video to Gallery: {str(e)}")
            # We re-raise to alert the user if this critical step fails
            raise e

        # Construct response data
        response_data = {
            "code": "success",
            "task_id": task_id,
            "prompt": self.prompt,
            "model": self.model,
            "video_url": video_url,
            "images_count": len(images_base64),
            "gallery_video_name": video_name
        }

        return Veo3NodeOutput(
            video_url=video_url,
            response_json=json.dumps(response_data),
            video_name=video_name,
            video=ImageField(image_name=video_name) # Pass video as ImageField to trigger frontend refresh
        )