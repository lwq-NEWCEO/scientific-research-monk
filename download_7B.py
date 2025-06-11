# download_deepseek_7b_chat.py
from huggingface_hub import snapshot_download
import os

# 定义模型信息
model_repo_id = "deepseek-ai/deepseek-llm-7b-chat"
# 定义本地保存目录，通常与模型ID对应，放在您期望的位置
# 例如，保存在 E:\DesignThinking\model\7B
local_download_dir = r"E:\DesignThinking\model\7B"

print(f"开始下载模型: {model_repo_id}")
print(f"将保存到本地目录: {local_download_dir}")

# 确保本地目录存在
os.makedirs(local_download_dir, exist_ok=True)

try:
    snapshot_download(
        repo_id=model_repo_id,
        local_dir=local_download_dir,
        revision="main", # 通常下载最新的 main 分支
        resume_download=True, # 支持断点续传
        local_dir_use_symlinks=False # 在 Windows 上建议设置为 False
    )
    print(f"\n模型 {model_repo_id} 下载完成！")

except Exception as e:
    print(f"\n下载模型 {model_repo_id} 时发生错误: {e}")
    print("请检查网络连接，或者是否有权限写入到指定的本地目录。")
    print("如果问题持续存在，尝试删除本地目录重新下载，或检查Hugging Face Hub状态。")

