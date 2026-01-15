import os
import subprocess
import sys
import time

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "invokeai"))
UPSTREAM_URL = "https://github.com/invoke-ai/InvokeAI.git"

def run_git(args, cwd=REPO_DIR):
    try:
        # Prevent git from opening a pager
        env = os.environ.copy()
        env["GIT_PAGER"] = "cat"
        
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            env=env
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Don't print error for status checks that are expected to fail sometimes
        return None

def main():
    # Ensure init has happened
    if not os.path.exists(os.path.join(REPO_DIR, ".git")):
        print("首次运行，正在初始化更新系统... (First run, initializing update system...)")
        import sys; subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "init_repo.py")])
    
    print("正在检查 InvokeAI 官方更新... (Checking for official InvokeAI updates...)")
    
    # Fetch upstream
    run_git(["fetch", "upstream", "main"])
    
    # Get hash of upstream/main and HEAD
    upstream_sha = run_git(["rev-parse", "upstream/main"])
    local_sha = run_git(["rev-parse", "HEAD"])
    
    # Try to get version info
    local_version = "Unknown"
    try:
        import sys
        if REPO_DIR not in sys.path:
            sys.path.append(REPO_DIR)
        from invokeai.version.invokeai_version import __version__
        local_version = __version__
    except Exception:
        pass

    # Get remote version from invokeai/version/invokeai_version.py in upstream
    remote_version = "Unknown"
    try:
        # git show upstream/main:invokeai/version/invokeai_version.py
        content = run_git(["show", "upstream/main:invokeai/version/invokeai_version.py"])
        if content:
            for line in content.splitlines():
                if "__version__" in line:
                    remote_version = line.split("=")[1].strip().strip('"').strip("'")
                    break
    except Exception:
        pass

    # Check if we have merged upstream/main before
    # simple check: is upstream/main reachable from HEAD?
    # If not, we might have updates.
    
    # Since we are doing a merge workflow, we want to know if upstream has commits we don't have.
    # git log HEAD..upstream/main --oneline
    changes = run_git(["log", "HEAD..upstream/main", "--oneline", "-n", "5"])
    
    if not changes:
        print(f"当前版本: {local_version} (已是最新) (Current version: {local_version} - Already up to date)")
        return

    print("\n" + "="*50)
    print("发现官方新版本！ (New official version found!)")
    print(f"本地版本: {local_version} (Local version)")
    print(f"官方版本: {remote_version} (Official version)")
    print("-" * 50)
    print("最新变更 (前5条) (Latest changes - Top 5):")
    print(changes)
    print("="*50)
    
    user_input = input("\n是否要尝试合并更新？(保留您的修改) [Y/n] (Do you want to try merging updates? Keep your changes): ").strip().lower()
    
    if user_input == 'y' or user_input == '':
        print("\n开始合并更新... (Starting update merge...)")
        print("注意：如果有文件冲突，系统将保留您的版本，并生成冲突标记。 (Note: If there are file conflicts, the system will keep your version and generate conflict markers.)")
        
        # Using --allow-unrelated-histories for the first merge if needed
        # But standard merge should be fine after init.
        # Actually, if we initialized with 'git init', upstream is unrelated.
        
        cmd = ["merge", "upstream/main", "--allow-unrelated-histories"]
        
        try:
            result = subprocess.run(
                ["git"] + cmd,
                cwd=REPO_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                print("\n✅ 更新合并成功！ (Update merged successfully!)")
            else:
                print("\n⚠️ 合并过程中遇到冲突 (这在魔改版中很正常)。 (Conflicts encountered during merge - Normal for modified versions.)")
                print("Git 已尝试自动合并。请检查代码中的冲突标记 (<<<<<<<)。 (Git attempted automatic merge. Please check code for conflict markers.)")
                print("或者，您可以手动解决冲突。 (Or you can resolve conflicts manually.)")
                print(f"Git 输出 (Git Output): {result.stdout}")
        except Exception as e:
            print(f"合并出错: {e} (Merge error: {e})")
            
        print("\n按回车键继续启动 InvokeAI... (Press Enter to continue starting InvokeAI...)")
        input()
    else:
        print("跳过更新。 (Update skipped.)")

if __name__ == "__main__":
    main()