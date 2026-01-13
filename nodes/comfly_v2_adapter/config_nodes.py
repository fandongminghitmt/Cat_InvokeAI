import os
import json
from typing import Literal, Optional
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, InvocationContext
from invokeai.app.invocations.fields import InputField, UIType
from invokeai.app.invocations.primitives import StringOutput

baseurl = "https://ai.comfly.chat"

def get_config():
    try:
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Comflyapi.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

@invocation("comfly_api_config", title="Comfly API Config", tags=["config", "api", "key"], category="comfly", version="1.0.0")
class ComflyAPIConfigInvocation(BaseInvocation):
    """Set Comfly API Key and Base URL configuration"""
    
    api_key: str = InputField(description="API Key to save/use")
    api_base: Literal["comfly", "ip", "hk", "us"] = InputField(default="comfly", description="Select API Line (comfly/hk/us/ip)")
    custom_ip: str = InputField(default="", description="Custom IP/URL (required if api_base='ip')")
    
    def invoke(self, context: InvocationContext) -> StringOutput:
        global baseurl
        
        # Determine Base URL
        base_url_mapping = {
            "comfly": "https://ai.comfly.chat",
            "hk": "https://hk-api.gptbest.vip",
            "us": "https://api.gptbest.vip",
            "ip": self.custom_ip.strip()
        }
        
        target_url = base_url_mapping.get(self.api_base, "https://ai.comfly.chat")
        
        if self.api_base == "ip" and not target_url:
             # If IP selected but no IP provided, fallback or warn?
             # For now, let's just fallback to default if empty
             target_url = "https://ai.comfly.chat"

        # Update global variable
        baseurl = target_url
        
        # Save to config file
        config = get_config()
        if self.api_key.strip():
            config['api_key'] = self.api_key
        
        # Always save the selected base_url so other nodes can pick it up
        config['base_url'] = target_url
        
        save_config(config)
            
        return StringOutput(value=self.api_key)
