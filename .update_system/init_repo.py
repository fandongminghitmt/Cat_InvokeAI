import os
import subprocess
import sys

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "invokeai"))
UPSTREAM_URL = "https://github.com/invoke-ai/InvokeAI.git"

def run_git(args, cwd=REPO_DIR):
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8' # Ensure encoding handles Chinese output if any
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        print(f"Stderr: {e.stderr}")
        return None

def init_git():
    print("正在初始化增量更新系统... (Initializing incremental update system...)")
    
    if not os.path.exists(os.path.join(REPO_DIR, ".git")):
        print(f"在 {REPO_DIR} 初始化 Git 仓库... (Initializing Git repository at {REPO_DIR}...)")
        run_git(["init"])
        
        # Configure local user for this repo
        run_git(["config", "user.name", "Cat_InvokeAI_User"])
        run_git(["config", "user.email", "local@update.system"])
        
        # Add all current files
        print("正在建立当前代码基准快照（这可能需要几秒钟）... (Creating current code baseline snapshot - this may take a few seconds...)")
        run_git(["add", "."])
        run_git(["commit", "-m", "Initial Baseline: Local Modified Version"])
        
        # Add upstream
        print(f"添加官方源: {UPSTREAM_URL} (Adding upstream source)")
        run_git(["remote", "add", "upstream", UPSTREAM_URL])
        
        print("初始化完成！ (Initialization complete!)")
    else:
        print("Git 仓库已存在，跳过初始化。 (Git repository already exists, skipping initialization.)")

    # Fetch latest info
    print("正在获取官方最新版本信息... (Fetching latest official version info...)")
    run_git(["fetch", "upstream", "main"])

if __name__ == "__main__":
    init_git()