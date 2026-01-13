import base64
import io
import os
import requests
import shutil
import cv2
import tempfile
from PIL import Image
from typing import Optional, Tuple

def pil_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_pil(base64_str: str) -> Image.Image:
    if 'base64,' in base64_str:
        base64_str = base64_str.split('base64,')[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

class ComflyVideoAdapter:
    def __init__(self, video_path_or_url: str):
        if video_path_or_url.startswith('http'):
            self.is_url = True
            self.video_url = video_path_or_url
            self.video_path = None
        else:
            self.is_url = False
            self.video_path = video_path_or_url
            self.video_url = None
        
    def get_dimensions(self) -> Tuple[int, int]:
        if self.is_url:
            return 1280, 720
        else:
            try: 
                cap = cv2.VideoCapture(self.video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return width, height
            except Exception as e:
                print(f'Error getting video dimensions: {str(e)}')
                return 1280, 720
            
    def save_to(self, output_path: str) -> bool:
        if self.is_url:
            try:
                response = requests.get(self.video_url, stream=True)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except Exception as e:
                print(f'Error downloading video from URL: {str(e)}')
                return False
        else:
            try:
                shutil.copyfile(self.video_path, output_path)
                return True
            except Exception as e:
                print(f'Error saving video: {str(e)}')
                return False


def find_or_create_board(context, board_name_or_id: Optional[str]) -> Optional[str]:
    """
    Finds a board by ID or name. If not found by ID, searches by name.
    If still not found, creates a new board with the given name.
    Returns the board_id, or None if input is empty/None.
    """
    if not board_name_or_id:
        return None
    
    board_name_or_id = board_name_or_id.strip()
    if not board_name_or_id:
        return None

    # Get all boards to check names and IDs
    try:
        all_boards = context.boards.get_all()
    except Exception as e:
        print(f"Error getting boards: {e}")
        return None
    
    # Check if any board matches the ID
    for board in all_boards:
        if board.board_id == board_name_or_id:
            return board.board_id
            
    # Check if any board matches the name
    for board in all_boards:
        if board.board_name == board_name_or_id:
            return board.board_id
            
    # If not found, create it
    try:
        new_board = context.boards.create(board_name_or_id)
        return new_board.board_id
    except Exception as e:
        print(f"Failed to create board '{board_name_or_id}': {e}")
        return None
