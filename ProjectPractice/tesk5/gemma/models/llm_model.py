"""LLM模型加载和推理"""
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_PATH


@st.cache_resource
def load_model():
    """加载模型，使用transformer"""
    try:
        if torch.cuda.is_available():
            torch_dtype = torch.float16
            print(f"[INFO] 检测到 GPU，使用 float16 加速")
        else:
            torch_dtype = torch.float32
            print(f"[INFO] 未检测到 GPU，使用 CPU 模式")

        print(f"[INFO] 正在加载 tokenizer: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )

        print(f"[INFO] 正在加载模型: {MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        print("[SUCCESS] 模型加载成功！")
        return model, tokenizer
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {type(e).__name__}")
        print(f"[ERROR] 详细错误: {str(e)}")
        print(f"[ERROR] 模型路径: {MODEL_PATH}")
        import traceback
        print(f"[ERROR] 完整堆栈:\n{traceback.format_exc()}")
        return None, None


def load_prompt_template():
    """从prompt.txt加载提示词模板"""
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return """你是一个专业的旅游推荐助手。你的职责是：
                    1. 根据用户的地貌偏好推荐合适的旅游目的地
                    2. 回答用户关于推荐目的地的问题，包括景点、美食、交通、住宿等
                    3. 提供实用的旅游建议和注意事项
                    4. 保持友好、专业、简洁的态度

                    {destination_context}

                    请围绕旅游推荐主题简洁回答，每次回答控制在200字以内。"""
