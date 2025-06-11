# verify_model.py - 尝试加载完整模型并进行简单推理

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch # 导入 torch 用于检查 cuda 可用性

# !!! 请将这个路径修改为您在本地Windows上实际存放 DeepSeek 模型文件的路径 !!!
model_path = "E:/DesignThinking/model/7B"

print(f"正在尝试从本地路径加载模型和分词器: {model_path}")

try:
    # --- 基础路径和文件存在性检查 ---
    if not os.path.exists(model_path):
        print(f"错误：指定的模型路径不存在 - {model_path}")
        exit() # 如果路径不存在，直接退出

    if not os.path.isdir(model_path):
         print(f"错误：指定的模型路径不是一个目录 - {model_path}")
         exit() # 如果不是目录，直接退出

    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
         print(f"错误：在模型路径 '{model_path}' 中未找到 'config.json' 文件。")
         print("这通常意味着模型没有正确下载或路径设置错误。")
         exit() # config.json 是核心文件，没有它模型无法加载

    # --- 尝试加载分词器 ---
    print("尝试加载分词器...")
    # DeepSeek 和 Qwen 等模型可能需要 trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("分词器加载成功！")

    # --- 尝试加载模型配置 (不加载全部权重，速度快) ---
    print("尝试加载模型配置...")
    # trust_remote_code=True 同样可能需要用于加载模型配置
    # low_cpu_mem_usage=True 有助于在加载大型模型时管理内存
    config = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto", # 尝试自动选择合适的浮点类型
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).config # 只获取配置，不加载全部权重
    print("模型配置加载成功！")
    print(f"模型类型: {config.model_type}")
    # 打印一些配置信息作为额外验证
    print(f"隐藏层数量: {config.num_hidden_layers}")
    print(f"隐藏层维度: {config.hidden_size}")
    if hasattr(config, 'num_attention_heads'):
        print(f"注意力头数量: {config.num_attention_heads}")

    # --- 尝试加载完整模型 (需要更多资源) ---
    print("\n尝试加载完整模型 (需要更多资源)...")
    # 自动选择设备：如果有 CUDA GPU，就用 'cuda'，否则用 'cpu'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"将模型加载到: {device}")

    print("开始加载完整模型权重...")
    full_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto", # 尝试自动选择 dtype (bfloat16, float16, float32)
        trust_remote_code=True, # 对于 DeepSeek/Qwen 这类模型可能需要
        low_cpu_mem_usage=True # 尝试减少 CPU 内存使用，尤其在加载到 CPU 时
    ).to(device) # 将模型移动到选定的设备

    print("完整模型加载成功！")
    print(f"模型已加载到: {device}")
    # 打印模型占用的显存/内存信息 (如果加载到 GPU)
    if device == "cuda":
        print(f"GPU 内存使用情况: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB (max)")


    # --- 进行一个简单的推理检查，确保模型可用 ---
    print("\n尝试进行简单推理...")
    try:
        # 对于聊天模型，需要构建 chat 历史并使用 apply_chat_template
        # DeepSeek 模型通常使用类似 ChatML 或 Alpaca 的格式，
        # apply_chat_template 会处理好这些细节。
        chat_history = [
            {"role": "user", "content": "你好，请介绍一下你自己。"} # 示例 prompt
            # 您可以添加更多对话轮次，但这只是一个简单测试
            # {"role": "assistant", "content": "我是一个大型语言模型..."}
            # {"role": "user", "content": "好的，那么..."}
        ]

        # 使用 tokenizer 的 apply_chat_template 生成模型所需的输入格式
        # tokenize=True 会直接返回 token id 张量
        # add_generation_prompt=True 会在用户输入后添加助手的起始标记，引导模型生成
        input_ids = tokenizer.apply_chat_template(
            chat_history,
            tokenize=True, # 返回张量
            add_generation_prompt=True, # 添加助手提示标记
            return_tensors="pt" # 返回 PyTorch 张量
        ).to(device) # 将输入张量移动到与模型相同的设备

        print(f"Prompt (通过 Chat Template 构建): {chat_history}")
        print(f"Encoded input_ids shape: {input_ids.shape}")


        # 生成响应
        print("开始生成响应...")
        # 可以调整生成参数以获得不同风格的输出
        output_ids = full_model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 200, # 最多生成输入长度 + 200 个 token
            num_return_sequences=1, # 只生成一个序列
            do_sample=True, # 启用采样，让输出更自然
            top_p=0.8, # 使用 Top-P 采样
            temperature=0.7, # 采样温度
            eos_token_id=tokenizer.eos_token_id, # 指定结束标记
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, # 指定 padding 标记 (有时需要)
        )
        print("响应生成完成！")

        # 解码生成的 token
        # output_ids 会包含输入的 prompt token，我们需要只解码生成的部分
        # 简单的做法是解码整个 output_ids，然后去掉 prompt 部分（这取决于 chat template 如何处理）
        # 更安全的方法是找到输入部分的长度，然后从那里开始解码，或者直接解码整个 output_ids
        generated_text_with_prompt = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # DeepSeek chat template通常会将 assistant response 直接跟在 prompt 后面
        # 找到 assistant 回复的起始位置通常比较复杂，简单起见，我们可以尝试找到最后一个用户/系统标记之后的部分
        # 或者先解码整个，然后判断是不是包含了完整的输入 prompt，再进行截断。
        # 对于简单的验证，我们直接解码整个输出并打印
        print(f"Generated Response (包含 Prompt): \n{generated_text_with_prompt}")

        # 如果想只获取生成的部分，可能需要更复杂的逻辑，取决于模型的 chat template 行为
        # print("\n注意：上述生成结果可能包含原始 prompt，如需仅获取生成部分，需解析模型的 Chat Template 格式。")


    except Exception as inference_e:
        print(f"\n进行简单推理时发生错误: {inference_e}")
        print("错误详情:", inference_e)
        print("这可能是由于：")
        print("- 显存/内存仍然不足 (即使加载成功，推理过程也需要额外资源)。")
        print("- 推理参数设置问题 (如 max_length 过大)。")
        print("- 模型或分词器的内部问题。")
        print(f"错误类型: {type(inference_e).__name__}")


    print("\n模型验证过程结束。")

except Exception as e:
    print(f"\n加载模型时发生错误: {e}")
    print("错误详情:", e)
    print("\n加载完整模型或分词器时发生错误。这可能是由于：")
    print("- 模型文件不完整或损坏（请再次检查目录文件）。")
    print("- **显存不足**：14B 模型需要大量显存（通常 20GB 或更多）。如果显存不足，尝试加载到 'cuda' 设备会报 CUDA Out of Memory (OOM) 错误。")
    print("- **CPU 内存不足**：如果加载到 'cpu' 且模型文件较大，可能需要大量内存。")
    print("- CUDA 安装或配置问题 (如果尝试加载到 'cuda')。")
    print("- 模型路径是否正确？")
    print("- `rag_env` 环境是否已激活并安装了所有必需的库 (transformers, torch, etc.)？")
    print("- 对于 DeepSeek/Qwen 模型，是否需要 trust_remote_code=True？ (已包含在脚本中)")
    print(f"错误类型: {type(e).__name__}")
    print("请根据错误信息排查。如果是显存不足，您可能需要更强大的 GPU，或者考虑使用量化版本的模型 (如 4-bit)。")

