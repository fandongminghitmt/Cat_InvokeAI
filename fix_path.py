import os
import sys

def fix_venv_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # The portable python executable location
    python_home = os.path.join(base_dir, 'uv', 'python', 'cpython-3.12.9-windows-x86_64-none')
    venv_cfg = os.path.join(base_dir, '.venv', 'pyvenv.cfg')
    
    if not os.path.exists(python_home):
        print(f"Error: Python home not found at {python_home}")
        return

    if not os.path.exists(venv_cfg):
        print(f"Error: venv config not found at {venv_cfg}")
        return
        
    # Read current config
    try:
        with open(venv_cfg, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading pyvenv.cfg: {e}")
        return
        
    new_lines = []
    changed = False
    
    # Check if update is needed
    for line in lines:
        if line.strip().startswith('home ='):
            current_path = line.strip()[6:].strip()
            # If path matches (ignoring case and trailing slash), no need to update
            if current_path.lower().rstrip('\\/') == python_home.lower().rstrip('\\/'):
                # print("Path is already correct.")
                new_lines.append(line)
            else:
                new_lines.append(f'home = {python_home}\n')
                changed = True
        else:
            new_lines.append(line)
            
    if changed:
        print(f"Updating pyvenv.cfg home to: {python_home}")
        try:
            with open(venv_cfg, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"Error writing pyvenv.cfg: {e}")

if __name__ == '__main__':
    try:
        fix_venv_config()
    except Exception as e:
        print(f"Failed to fix venv config: {e}")
